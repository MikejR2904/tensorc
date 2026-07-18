/// codegen/tools/test_real_execution.cpp
///
/// Real execution tests: real TensorC *source text* compiled through the
/// exact pipeline cli/tensorc.cpp uses (Lexer -> Parser -> SemanticAnalyzer
/// -> IRBuilder -> PassPipeline -> ScalarCodegenPipeline), assembled and
/// linked with the local x86_64-w64-mingw32 toolchain, and *executed* — the
/// result is compared against an independently-computed expected value.
///
/// This supersedes the original REAL_EXECUTION_TESTING_GUIDE.md draft, which
/// sketched this exact idea against an invented API that never matched the
/// real compiler (`Lexer::tokenize`, `IRBuilder::build(ast)` with no module,
/// `CodegenDriver::generate_assembly`) and was never implemented. See
/// codegen/tools/CompilationTestUtils.h for the real `compile_source_to_asm`
/// this file is built on, and codegen/tools/ExecUtils.h for the assemble
/// + link + load step.
///
/// Source-level tests here complement (not replace) codegen/tools/
/// test_scalar_pipeline.cpp's hand-built-IR tests: those give precise
/// control over specific MIR-level patterns (critical edges, register
/// pressure); these prove the *whole* pipeline — including the parser and
/// IRBuilder — end to end on programs a user would actually write. Several
/// of the bugs this file's first real run caught were in fact frontend bugs
/// invisible to hand-built-IR testing (see git history around this file's
/// introduction): a null-callee segfault on unresolved module calls, a
/// loop-body variable-staleness bug from lazy alloca promotion, and a
/// function-symbol naming mismatch between call sites and definitions.
///
/// Scope: scalar/control-flow TensorC programs only (matches
/// ScalarCodegenPipeline's scope). Programs using tensor ops or builtin
/// modules (math::, tensor::, ...) are out of scope — those lower to calls
/// against mangled names like "math.sin" with no runtime symbol behind them
/// yet, so they compile but can't link; see codegen/README.md's "Known gaps".

#include <gtest/gtest.h>
#include "CompilationTestUtils.h"
#include "ExecUtils.h"
#include <cmath>
#include <cstdint>
#include <string>

using codegen::testing::assemble_and_load;
using codegen::testing::compile_source_to_asm;
using codegen::testing::get_symbol;
using codegen::testing::unload;

// Named aliases for every signature these tests call — GCC's parser rejects
// __attribute__((sysv_abi)) inside an inline function-pointer type used
// directly as an explicit template argument (`resolve<int32_t(*)(int32_t)
// __attribute__((sysv_abi))>` fails to parse), so the attribute has to be
// attached via a typedef first, same as codegen/tools/test_scalar_pipeline.cpp.
typedef int32_t (*sysv_i32_i32)(int32_t) __attribute__((sysv_abi));
typedef int32_t (*sysv_i32_i32i32)(int32_t, int32_t) __attribute__((sysv_abi));
typedef int64_t (*sysv_i64_i64i64)(int64_t, int64_t) __attribute__((sysv_abi));
typedef double  (*sysv_f64_f64f64)(double, double) __attribute__((sysv_abi));

namespace {

int g_counter = 0;
std::string scratch_prefix() { return "real_exec_scratch_" + std::to_string(g_counter++); }

/// Compiles `source`, assembles+links+loads it, resolves `symbol`, and
/// returns the callable function pointer cast to Fn — or fails the current
/// test and returns nullptr. Fn must carry __attribute__((sysv_abi)) so the
/// local MinGW-w64 toolchain (Windows x64 calling convention by default)
/// calls into the generated System V AMD64 code correctly.
template <typename Fn>
Fn resolve(const std::string& source, const std::string& symbol, void** out_module) {
    // EXPECT_ (not ASSERT_) NO_THROW: ASSERT_* expands to a bare `return;` on
    // failure, which only compiles in a void-returning function — resolve()
    // returns Fn, so it has to record-and-continue instead.
    std::string asm_text;
    EXPECT_NO_THROW(asm_text = compile_source_to_asm(source, "x86_64")) << "compilation failed";
    if (asm_text.empty()) return nullptr;
    void* mod = assemble_and_load(asm_text, scratch_prefix());
    EXPECT_NE(mod, nullptr) << "assemble/link failed for:\n" << asm_text;
    if (!mod) return nullptr;
    *out_module = mod;
    Fn f = reinterpret_cast<Fn>(get_symbol(mod, symbol));
    EXPECT_NE(reinterpret_cast<void*>(f), nullptr) << "symbol '" << symbol << "' not found";
    return f;
}

} // namespace

// ── i32 arithmetic (the width most likely to expose register-size bugs,
//    since function args/locals default to i32 and memory loads/stores are
//    narrower than the 64-bit GPRs they get loaded into) ───────────────────

TEST(RealExecutionTest, I32Add) {
    void* mod = nullptr;
    auto f = resolve<sysv_i32_i32i32>(
        "fn add_i32(a: i32, b: i32) -> i32 { return a + b; }", "add_i32", &mod);
    ASSERT_NE(f, nullptr);
    EXPECT_EQ(f(17, 25), 42);
    EXPECT_EQ(f(-5, 5), 0);
    unload(mod);
}

TEST(RealExecutionTest, I32SubMulDivMod) {
    struct Case { const char* op; const char* name; int32_t a, b, expected; };
    // No `%` (modulo) case: TensorC's grammar has no modulo operator token
    // (compiler/lexer, compiler/parser) even though ir::BinOpCode::Mod
    // exists in the IR — there's no surface syntax that reaches it, so it's
    // not something this source-level test can exercise.
    Case cases[] = {
        {"-", "sub_i32", 50, 8, 42},
        {"*", "mul_i32", 6, 7, 42},
        {"/", "div_i32", 100, 3, 33},
    };
    for (auto& c : cases) {
        std::string src = std::string("fn ") + c.name + "(a: i32, b: i32) -> i32 { return a " + c.op + " b; }";
        void* mod = nullptr;
        auto f = resolve<sysv_i32_i32i32>(src, c.name, &mod);
        ASSERT_NE(f, nullptr) << c.name;
        EXPECT_EQ(f(c.a, c.b), c.expected) << c.name;
        unload(mod);
    }
}

// ── i64 arithmetic ───────────────────────────────────────────────────────

TEST(RealExecutionTest, I64Add) {
    void* mod = nullptr;
    auto f = resolve<sysv_i64_i64i64>(
        "fn add_i64(a: i64, b: i64) -> i64 { return a + b; }", "add_i64", &mod);
    ASSERT_NE(f, nullptr);
    EXPECT_EQ(f(1000000000000LL, 2000000000000LL), 3000000000000LL);
    unload(mod);
}

// ── f64 arithmetic ───────────────────────────────────────────────────────

TEST(RealExecutionTest, F64Add) {
    void* mod = nullptr;
    auto f = resolve<sysv_f64_f64f64>(
        "fn add_f64(a: f64, b: f64) -> f64 { return a + b; }", "add_f64", &mod);
    ASSERT_NE(f, nullptr);
    EXPECT_DOUBLE_EQ(f(1.5, 2.25), 3.75);
    unload(mod);
}

// ── comparisons + conditional branches ───────────────────────────────────

TEST(RealExecutionTest, MaxViaIfElse) {
    void* mod = nullptr;
    auto f = resolve<sysv_i32_i32i32>(
        "fn max_i32(a: i32, b: i32) -> i32 { if (a > b) { return a; } else { return b; } }", "max_i32", &mod);
    ASSERT_NE(f, nullptr);
    EXPECT_EQ(f(17, 42), 42);
    EXPECT_EQ(f(99, 3), 99);
    EXPECT_EQ(f(5, 5), 5);
    unload(mod);
}

TEST(RealExecutionTest, BooleanLogic) {
    void* mod = nullptr;
    auto f = resolve<sysv_i32_i32i32>(
        "fn in_range(x: i32, n: i32) -> i32 { if (x >= 0 && x <= n) { return 1; } else { return 0; } }",
        "in_range", &mod);
    ASSERT_NE(f, nullptr);
    EXPECT_EQ(f(5, 10), 1);
    EXPECT_EQ(f(-1, 10), 0);
    EXPECT_EQ(f(11, 10), 0);
    unload(mod);
}

// ── loops: this is the exact pattern (accumulator read-and-written inside
//    the loop body) that used to be silently miscompiled — see lower_let's
//    doc comment in compiler/ir/IRBuilder.h. ─────────────────────────────

TEST(RealExecutionTest, WhileLoopSumToN) {
    void* mod = nullptr;
    auto f = resolve<sysv_i32_i32>(
        R"(fn sum_to_n(n: i32) -> i32 {
            let i = 0;
            let s = 0;
            while (i <= n) {
                s = s + i;
                i = i + 1;
            }
            return s;
        })",
        "sum_to_n", &mod);
    ASSERT_NE(f, nullptr);
    EXPECT_EQ(f(10), 55);
    EXPECT_EQ(f(0), 0);
    EXPECT_EQ(f(100), 5050);
    unload(mod);
}

TEST(RealExecutionTest, WhileLoopFactorial) {
    void* mod = nullptr;
    auto f = resolve<sysv_i32_i32>(
        R"(fn factorial(n: i32) -> i32 {
            let i = 1;
            let result = 1;
            while (i <= n) {
                result = result * i;
                i = i + 1;
            }
            return result;
        })",
        "factorial", &mod);
    ASSERT_NE(f, nullptr);
    EXPECT_EQ(f(5), 120);
    EXPECT_EQ(f(0), 1);
    EXPECT_EQ(f(7), 5040);
    unload(mod);
}

TEST(RealExecutionTest, WhileLoopFibonacci) {
    // Two loop-carried accumulators updated from each other in the same
    // iteration — exercises reading both *before* either is overwritten.
    void* mod = nullptr;
    auto f = resolve<sysv_i32_i32>(
        R"(fn fib(n: i32) -> i32 {
            let a = 0;
            let b = 1;
            let i = 0;
            while (i < n) {
                let next = a + b;
                a = b;
                b = next;
                i = i + 1;
            }
            return a;
        })",
        "fib", &mod);
    ASSERT_NE(f, nullptr);
    EXPECT_EQ(f(0), 0);
    EXPECT_EQ(f(1), 1);
    EXPECT_EQ(f(10), 55);
    unload(mod);
}

// ── recursion (exercises the function-symbol-name fix: a call site and its
//    own definition must agree on a name once the "@" IR sigil is stripped) ─

TEST(RealExecutionTest, RecursiveFactorial) {
    void* mod = nullptr;
    auto f = resolve<sysv_i32_i32>(
        R"(fn factorial_rec(n: i32) -> i32 {
            if (n <= 1) {
                return 1;
            } else {
                return n * factorial_rec(n - 1);
            }
        })",
        "factorial_rec", &mod);
    ASSERT_NE(f, nullptr);
    EXPECT_EQ(f(5), 120);
    EXPECT_EQ(f(1), 1);
    EXPECT_EQ(f(0), 1);
    unload(mod);
}

// ── multi-function programs (real inter-procedural calls from real source,
//    not hand-built IR with hand-picked names) ───────────────────────────

TEST(RealExecutionTest, MultiFunctionCallChain) {
    void* mod = nullptr;
    auto f = resolve<sysv_i32_i32>(
        R"(
        fn square(x: i32) -> i32 { return x * x; }
        fn double_it(x: i32) -> i32 { return x + x; }
        fn compute(x: i32) -> i32 { return double_it(square(x)); }
        )",
        "compute", &mod);
    ASSERT_NE(f, nullptr);
    EXPECT_EQ(f(3), 18);  // square(3)=9, double_it(9)=18
    EXPECT_EQ(f(5), 50);  // square(5)=25, double_it(25)=50
    unload(mod);
}
