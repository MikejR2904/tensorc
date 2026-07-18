/// codegen/tools/test_scalar_pipeline.cpp
///
/// Execution-based verification for the new ScalarCodegenPipeline
/// (GenericISel -> PhiElimination -> CallLowering -> target selection ->
/// LinearScanRegAlloc -> PrologueEpilogInserter -> AsmPrinter), covering the
/// exact gaps the backend redesign plan identified in the old
/// codegen/legacy pipeline: real liveness-based register allocation,
/// correct phi elimination across critical edges, and a real ABI.
///
/// x86-64: assembled, linked, and *actually executed* via the local
/// x86_64-w64-mingw32 toolchain (see codegen/tools/ExecUtils.h) — numeric
/// results are checked against real answers, not just "does the text
/// contain an add mnemonic".
/// riscv64: no cross-assembler/emulator is available in this environment,
/// so it gets structural assertions only (this asymmetry is intentional and
/// documented in the redesign plan, not accidental).

#include "../../compiler/ir/IRModule.h"
#include "../../compiler/ir/Instruction.h"
#include "../../compiler/ir/IRPasses.h"
#include "../ScalarCodegenPipeline.h"
#include "ExecUtils.h"
#include "test_utils.h"
#include <cassert>
#include <iostream>
#include <sstream>

using namespace ir;
using namespace codegen::testing;

namespace {

ValuePtr borrow(Value* v) { return ValuePtr(v, [](Value*) {}); }

std::string codegen_for(const std::string& target, Function* fn) {
    codegen::ScalarCodegenPipeline pipeline(target);
    assert(pipeline.valid() && "unknown target");
    std::ostringstream oss;
    bool ok = pipeline.lower_function(*fn, oss);
    assert(ok && "codegen should succeed");
    return oss.str();
}

int g_pass = 0, g_fail = 0;
void check(bool cond, const std::string& msg) {
    if (cond) { ++g_pass; std::cout << "  ok: " << msg << "\n"; }
    else { ++g_fail; std::cout << "  FAIL: " << msg << "\n"; }
}

typedef int64_t (*sysv_i64_i64i64)(int64_t, int64_t) __attribute__((sysv_abi));
typedef double  (*sysv_f64_f64f64)(double, double) __attribute__((sysv_abi));
typedef int64_t (*sysv_i64_i64)(int64_t) __attribute__((sysv_abi));
typedef int64_t (*sysv_i64_void)() __attribute__((sysv_abi));

void exec_check_i64_i64i64(const std::string& asm_text, const std::string& sym, int64_t a, int64_t b, int64_t expected, const std::string& label) {
    static int counter = 0;
    void* mod = assemble_and_load(asm_text, "scratch_x86_" + std::to_string(counter++));
    check(mod != nullptr, label + " (assembled+linked)");
    if (!mod) return;
    auto* f = reinterpret_cast<sysv_i64_i64i64>(get_symbol(mod, sym));
    check(f != nullptr, label + " (symbol found)");
    if (f) {
        int64_t got = f(a, b);
        check(got == expected, label + " (" + std::to_string(a) + "," + std::to_string(b) + ") = " + std::to_string(got) + " expected " + std::to_string(expected));
    }
    unload(mod);
}

} // namespace

// ── i64 arithmetic ───────────────────────────────────────────────────────

void test_i64_add() {
    std::cout << "\n=== i64 add ===\n";
    auto mod = std::make_shared<IRModule>("t_add");
    auto* fn = mod->add_function("add_i64", Type::fn({Type::i64(), Type::i64()}, Type::i64()));
    fn->create_entry();
    auto* a = fn->add_param("a", Type::i64());
    auto* b = fn->add_param("b", Type::i64());
    auto* c = fn->entry()->emit<BinOpInst>("c", Type::i64(), BinOpCode::Add, borrow(a), borrow(b));
    fn->entry()->emit<ReturnInst>(borrow(c));
    CFGPass::run_on(*mod);

    std::string x86 = codegen_for("x86_64", fn);
    exec_check_i64_i64i64(x86, "add_i64", 17, 25, 42, "add_i64");

    std::string rv = codegen_for("riscv64", fn);
    check(AssemblyValidator(rv).has_mnemonic("add"), "riscv64 add_i64 contains 'add'");
    check(AssemblyValidator(rv).has_mnemonic("ret"), "riscv64 add_i64 ends in ret");
}

void test_i64_sub_mul_div_rem() {
    std::cout << "\n=== i64 sub/mul/sdiv/srem ===\n";
    struct Case { BinOpCode op; const char* name; int64_t a, b, expected; };
    Case cases[] = {
        {BinOpCode::Sub, "sub_i64", 50, 8, 42},
        {BinOpCode::Mul, "mul_i64", 6, 7, 42},
        {BinOpCode::Div, "div_i64", 100, 3, 33},
        {BinOpCode::Mod, "rem_i64", 100, 3, 1},
    };
    for (auto& c : cases) {
        auto mod = std::make_shared<IRModule>(c.name);
        auto* fn = mod->add_function(c.name, Type::fn({Type::i64(), Type::i64()}, Type::i64()));
        fn->create_entry();
        auto* a = fn->add_param("a", Type::i64());
        auto* b = fn->add_param("b", Type::i64());
        auto* r = fn->entry()->emit<BinOpInst>("r", Type::i64(), c.op, borrow(a), borrow(b));
        fn->entry()->emit<ReturnInst>(borrow(r));
        CFGPass::run_on(*mod);
        std::string x86 = codegen_for("x86_64", fn);
        exec_check_i64_i64i64(x86, c.name, c.a, c.b, c.expected, c.name);
    }
}

// ── f64 arithmetic ───────────────────────────────────────────────────────

void test_f64_add() {
    std::cout << "\n=== f64 add ===\n";
    auto mod = std::make_shared<IRModule>("t_fadd");
    auto* fn = mod->add_function("fadd_f64", Type::fn({Type::f64(), Type::f64()}, Type::f64()));
    fn->create_entry();
    auto* a = fn->add_param("a", Type::f64());
    auto* b = fn->add_param("b", Type::f64());
    auto* c = fn->entry()->emit<BinOpInst>("c", Type::f64(), BinOpCode::FAdd, borrow(a), borrow(b));
    fn->entry()->emit<ReturnInst>(borrow(c));
    CFGPass::run_on(*mod);

    std::string x86 = codegen_for("x86_64", fn);
    void* mod_h = assemble_and_load(x86, "scratch_x86_fadd");
    check(mod_h != nullptr, "fadd_f64 assembled+linked");
    if (mod_h) {
        auto* f = reinterpret_cast<sysv_f64_f64f64>(get_symbol(mod_h, "fadd_f64"));
        check(f != nullptr, "fadd_f64 symbol found");
        if (f) { double got = f(1.5, 2.25); check(got == 3.75, "fadd_f64(1.5,2.25) = " + std::to_string(got)); }
        unload(mod_h);
    }
}

// ── comparison + conditional branch (max) ────────────────────────────────

void test_conditional_max() {
    std::cout << "\n=== conditional (max via phi) ===\n";
    auto mod = std::make_shared<IRModule>("t_max");
    auto* fn = mod->add_function("max_i64", Type::fn({Type::i64(), Type::i64()}, Type::i64()));
    auto* entry = fn->create_entry();
    auto* then_bb = fn->add_block("then");
    auto* else_bb = fn->add_block("else");
    auto* merge_bb = fn->add_block("merge");

    auto* a = fn->add_param("a", Type::i64());
    auto* b = fn->add_param("b", Type::i64());
    auto* cmp = entry->emit<CmpInst>("cmp", CmpCode::Gt, borrow(a), borrow(b));
    entry->emit<CondBranchInst>(borrow(cmp), then_bb, else_bb);

    then_bb->emit<BranchInst>(merge_bb);
    else_bb->emit<BranchInst>(merge_bb);

    auto* phi = merge_bb->emit<PhiInst>("result", Type::i64());
    phi->add_incoming(borrow(a), then_bb);
    phi->add_incoming(borrow(b), else_bb);
    merge_bb->emit<ReturnInst>(borrow(phi));

    CFGPass::run_on(*mod);
    std::string x86 = codegen_for("x86_64", fn);
    exec_check_i64_i64i64(x86, "max_i64", 17, 42, 42, "max_i64(17,42)");
    exec_check_i64_i64i64(x86, "max_i64", 99, 3, 99, "max_i64(99,3)");

    std::string rv = codegen_for("riscv64", fn);
    check(AssemblyValidator(rv).has_mnemonic("slt"), "riscv64 max_i64 contains 'slt'");
}

// ── critical-edge phi (A has 2 succs, merge has 3 preds) ─────────────────

void test_critical_edge_phi() {
    std::cout << "\n=== critical-edge phi elimination ===\n";
    auto mod = std::make_shared<IRModule>("t_crit");
    auto* fn = mod->add_function("crit_edge", Type::fn({Type::i64(), Type::i64()}, Type::i64()));
    auto* entry = fn->create_entry();
    auto* A = fn->add_block("A");
    auto* B = fn->add_block("B");
    auto* C = fn->add_block("C");
    auto* merge = fn->add_block("merge");

    auto* x = fn->add_param("x", Type::i64());
    auto* y = fn->add_param("y", Type::i64());
    auto* cond1 = entry->emit<CmpInst>("cond1", CmpCode::Gt, borrow(x), borrow(y));
    entry->emit<CondBranchInst>(borrow(cond1), A, B);

    // A has two successors (merge, C) — the A->merge edge is critical
    // because `merge` also has more than one predecessor.
    auto* cond2 = A->emit<CmpInst>("cond2", CmpCode::Gt, borrow(y), borrow(x));
    A->emit<CondBranchInst>(borrow(cond2), merge, C);
    B->emit<BranchInst>(merge);
    C->emit<BranchInst>(merge);

    auto* phi = merge->emit<PhiInst>("result", Type::i64());
    phi->add_incoming(borrow(x), A);
    phi->add_incoming(borrow(y), B);
    phi->add_incoming(borrow(y), C);
    merge->emit<ReturnInst>(borrow(phi));

    CFGPass::run_on(*mod);
    std::string x86 = codegen_for("x86_64", fn);
    // x < y  -> entry takes B -> merge gets y
    exec_check_i64_i64i64(x86, "crit_edge", 1, 5, 5, "crit_edge via B");
    // x > y, then y > x is false (since x>y) -> A takes C -> merge gets y
    exec_check_i64_i64i64(x86, "crit_edge", 5, 1, 1, "crit_edge via A->C");
}

// ── loop with back-edge phis (sum 1..n) ──────────────────────────────────

void test_loop_sum() {
    std::cout << "\n=== loop (sum 1..n) ===\n";
    auto mod = std::make_shared<IRModule>("t_loop");
    auto* fn = mod->add_function("sum_to_n", Type::fn({Type::i64()}, Type::i64()));
    auto* entry = fn->create_entry();
    auto* header = fn->add_block("header");
    auto* body = fn->add_block("body");
    auto* exit_bb = fn->add_block("exit");

    auto* n = fn->add_param("n", Type::i64());
    auto zero = std::make_shared<ConstantInt>(0, Type::i64());
    auto one = std::make_shared<ConstantInt>(1, Type::i64());
    entry->emit<BranchInst>(header);

    auto* i_phi = header->emit<PhiInst>("i", Type::i64());
    auto* sum_phi = header->emit<PhiInst>("sum", Type::i64());
    auto* cond = header->emit<CmpInst>("cond", CmpCode::Le, borrow(i_phi), borrow(n));
    header->emit<CondBranchInst>(borrow(cond), body, exit_bb);

    auto* sum_next = body->emit<BinOpInst>("sum_next", Type::i64(), BinOpCode::Add, borrow(sum_phi), borrow(i_phi));
    auto* i_next = body->emit<BinOpInst>("i_next", Type::i64(), BinOpCode::Add, borrow(i_phi), ValuePtr(one));
    body->emit<BranchInst>(header);

    i_phi->add_incoming(ValuePtr(zero), entry);
    i_phi->add_incoming(borrow(i_next), body);
    sum_phi->add_incoming(ValuePtr(zero), entry);
    sum_phi->add_incoming(borrow(sum_next), body);

    exit_bb->emit<ReturnInst>(borrow(sum_phi));

    CFGPass::run_on(*mod);
    std::string x86 = codegen_for("x86_64", fn);
    void* mod_h = assemble_and_load(x86, "scratch_x86_loop");
    check(mod_h != nullptr, "sum_to_n assembled+linked");
    if (mod_h) {
        auto* f = reinterpret_cast<sysv_i64_i64>(get_symbol(mod_h, "sum_to_n"));
        check(f != nullptr, "sum_to_n symbol found");
        if (f) {
            int64_t got = f(10);
            check(got == 55, "sum_to_n(10) = " + std::to_string(got) + " expected 55");
        }
        unload(mod_h);
    }
}

// ── register pressure (forces spills under real liveness) ────────────────

void test_register_pressure() {
    std::cout << "\n=== register pressure (spills) ===\n";
    auto mod = std::make_shared<IRModule>("t_pressure");
    auto* fn = mod->add_function("pressure", Type::fn({Type::i64()}, Type::i64()));
    fn->create_entry();
    auto* a = fn->add_param("a", Type::i64());
    ValuePtr cur = borrow(a);
    // 40 live-simultaneously temporaries: t_i = a + i, all kept alive by
    // being summed at the very end — forces real spill/reload code under
    // liveness-based allocation (there are only ~14 allocatable x86 GPRs).
    std::vector<ValuePtr> temps;
    for (int i = 0; i < 40; ++i) {
        auto imm = std::make_shared<ConstantInt>(i, Type::i64());
        auto* t = fn->entry()->emit<BinOpInst>("t" + std::to_string(i), Type::i64(), BinOpCode::Add, borrow(a), ValuePtr(imm));
        temps.push_back(borrow(t));
    }
    ValuePtr acc = temps[0];
    for (size_t i = 1; i < temps.size(); ++i) {
        auto* s = fn->entry()->emit<BinOpInst>("acc" + std::to_string(i), Type::i64(), BinOpCode::Add, acc, temps[i]);
        acc = borrow(s);
    }
    fn->entry()->emit<ReturnInst>(acc);
    CFGPass::run_on(*mod);

    std::string x86 = codegen_for("x86_64", fn);
    void* mod_h = assemble_and_load(x86, "scratch_x86_pressure");
    check(mod_h != nullptr, "pressure assembled+linked");
    if (mod_h) {
        auto* f = reinterpret_cast<sysv_i64_i64>(get_symbol(mod_h, "pressure"));
        check(f != nullptr, "pressure symbol found");
        if (f) {
            int64_t got = f(1);
            // sum_{i=0}^{39} (1 + i) = 40*1 + (0+..+39) = 40 + 780 = 820
            check(got == 820, "pressure(1) = " + std::to_string(got) + " expected 820");
        }
        unload(mod_h);
    }
}

// ── call between two generated functions (exercises CallLowering) ────────

void test_call_between_functions() {
    std::cout << "\n=== call lowering (caller -> callee) ===\n";
    auto mod = std::make_shared<IRModule>("t_call");
    auto* callee = mod->add_function("callee_double", Type::fn({Type::i64()}, Type::i64()));
    callee->create_entry();
    auto* p = callee->add_param("x", Type::i64());
    auto* doubled = callee->entry()->emit<BinOpInst>("doubled", Type::i64(), BinOpCode::Add, borrow(p), borrow(p));
    callee->entry()->emit<ReturnInst>(borrow(doubled));

    auto* caller = mod->add_function("caller_calls_double", Type::fn({Type::i64()}, Type::i64()));
    caller->create_entry();
    auto* a = caller->add_param("a", Type::i64());
    ValuePtr callee_val(callee, [](Value*) {});
    auto* call = caller->entry()->emit<CallInst>("r", Type::i64(), callee_val, std::vector<ValuePtr>{borrow(a)});
    caller->entry()->emit<ReturnInst>(borrow(call));

    CFGPass::run_on(*mod);

    codegen::ScalarCodegenPipeline pipeline("x86_64");
    std::ostringstream oss;
    pipeline.lower_function(*callee, oss);
    pipeline.lower_function(*caller, oss);
    std::string combined = oss.str();

    void* mod_h = assemble_and_load(combined, "scratch_x86_call");
    check(mod_h != nullptr, "caller/callee assembled+linked");
    if (mod_h) {
        auto* f = reinterpret_cast<sysv_i64_i64>(get_symbol(mod_h, "caller_calls_double"));
        check(f != nullptr, "caller_calls_double symbol found");
        if (f) {
            int64_t got = f(21);
            check(got == 42, "caller_calls_double(21) = " + std::to_string(got) + " expected 42");
        }
        unload(mod_h);
    }
}

int main() {
    test_i64_add();
    test_i64_sub_mul_div_rem();
    test_f64_add();
    test_conditional_max();
    test_critical_edge_phi();
    test_loop_sum();
    test_register_pressure();
    test_call_between_functions();

    std::cout << "\n===== " << g_pass << " passed, " << g_fail << " failed =====\n";
    return g_fail == 0 ? 0 : 1;
}
