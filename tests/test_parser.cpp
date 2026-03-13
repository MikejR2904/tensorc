#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>
#include <fstream>
#include <vector>
#include "../compiler/lexer/Lexer.h"
#include "../compiler/parser/Parser.h"
#include "../compiler/ast/ASTNode.h"

// ─────────────────────────────────────────────
//  ANSI colours
// ─────────────────────────────────────────────
#define CLR_RESET  "\033[0m"
#define CLR_GREEN  "\033[32m"
#define CLR_RED    "\033[31m"
#define CLR_CYAN   "\033[36m"
#define CLR_YELLOW "\033[33m"
#define CLR_BOLD   "\033[1m"

// ─────────────────────────────────────────────
//  AST printer
//  Walks the tree and prints it indented.
//  Mirrors Cortex's AstDebug trait.
// ─────────────────────────────────────────────

// forward declaration
class ASTPrinter {
public:
    std::ostream& out;
    ASTPrinter(std::ostream& os) : out(os) {}

    void print(const Program& program) {
        out << "Program\n";
        for (size_t i = 0; i < program.stmts.size(); ++i) {
            bool isLast = (i == program.stmts.size() - 1);
            printStmt(program.stmts[i].get(), "", isLast);
        }
    }

private:
    // Helper for your TyKind names
    std::string tyKindStr(TyKind t) {
        switch (t) {
            case TyKind::I32:    return "i32";
            case TyKind::I64:    return "i64";
            case TyKind::F32:    return "f32";
            case TyKind::F64:    return "f64";
            case TyKind::Bool:   return "bool";
            case TyKind::Str:    return "str";
            case TyKind::Void:   return "void";
            case TyKind::Tensor: return "Tensor";
            default:             return "Type";
        }
    }

    void printStmt(const Stmt* s, const std::string& prefix, bool isLast) {
        if (!s) return;

        out << prefix << (isLast ? "└── " : "├── ");
        const auto& k = s->kind;
        std::string nextPrefix = prefix + (isLast ? "    " : "│   ");

        switch (k.tag) {
            case StmtKind::Tag::Let:
                out << "Let(" << k.let_ident.name() << ": " << tyKindStr(k.let_ident.ty_kind()) << ")\n";
                if (k.let_expr) printExpr(k.let_expr.get(), nextPrefix, true);
                break;

            case StmtKind::Tag::Func:
                if (k.func.has_value()) {
                    const auto& f = k.func.value();
                    out << "Func(" << f.ident.name() << ") -> " << tyKindStr(f.ident.ty_kind()) << "\n";
                    for (size_t i = 0; i < f.params.size(); ++i) {
                        out << nextPrefix << "├── Param(" << f.params[i].name() << ")\n";
                    }
                    printCompound(f.body, nextPrefix, true);
                }
                break;

            case StmtKind::Tag::Return:
                out << "Return\n";
                if (k.ret_expr) printExpr(k.ret_expr.get(), nextPrefix, true);
                break;

            case StmtKind::Tag::If:
                out << "If\n";
                if (k.if_cond) printExpr(k.if_cond.get(), nextPrefix, false);
                printCompound(k.if_body, nextPrefix, k.else_or_else_if == nullptr);
                if (k.else_or_else_if) printStmt(k.else_or_else_if.get(), prefix, isLast); 
                break;

            case StmtKind::Tag::While:
                out << "While" << (k.while_cond ? "" : " (infinite)") << "\n";
                if (k.while_cond) printExpr(k.while_cond.get(), nextPrefix, false);
                printCompound(k.while_body, nextPrefix, true);
                break;

            case StmtKind::Tag::Struct:
                out << "StructDef [" << k.struct_name << "]\n";
                for (size_t i = 0; i < k.struct_fields.size(); ++i) {
                    bool lastF = (i == k.struct_fields.size() - 1);
                    out << nextPrefix << (lastF ? "└── " : "├── ") << "Field: " << k.struct_fields[i].name << "\n";
                }
                break;

            case StmtKind::Tag::Expr:
                out << "ExprStmt\n";
                if (k.expr) printExpr(k.expr.get(), nextPrefix, true);
                break;

            default:
                out << "<unimplemented stmt>\n";
                break;
        }
    }

    void printExpr(const Expr* e, const std::string& prefix, bool isLast) {
        if (!e) return;

        out << prefix << (isLast ? "└── " : "├── ");
        const auto& k = e->kind;
        std::string nextPrefix = prefix + (isLast ? "    " : "│   ");

        switch (k.tag) {
            case ExprKind::Tag::Lit:
                out << "Lit: " << k.lit.str_val << "\n";
                break;

            case ExprKind::Tag::Id:
                out << "Id: " << k.id.name() << " (" << tyKindStr(k.id.ty_kind()) << ")\n";
                break;

            case ExprKind::Tag::Binary:
                out << "Binary(" << (int)k.bin_op << ")\n"; // use your binOpStr here
                printExpr(k.lhs.get(), nextPrefix, false);
                printExpr(k.rhs.get(), nextPrefix, true);
                break;

            case ExprKind::Tag::Call:
                out << "Call\n";
                if (k.callee) printExpr(k.callee.get(), nextPrefix, k.args.empty());
                for (size_t i = 0; i < k.args.size(); ++i) {
                    printExpr(k.args[i].get(), nextPrefix, i == k.args.size() - 1);
                }
                break;

            case ExprKind::Tag::Field:
                out << "Field(." << k.member << ")\n";
                printExpr(k.target.get(), nextPrefix, true);
                break;

            default:
                out << "<unimplemented expr>\n";
                break;
        }
    }

    void printCompound(const Compound& c, const std::string& prefix, bool isLast) {
        std::string nextPrefix = prefix + (isLast ? "    " : "│   ");
        for (size_t i = 0; i < c.stmts.size(); ++i) {
            bool lastS = (i == c.stmts.size() - 1 && !c.tail_expr);
            printStmt(c.stmts[i].get(), nextPrefix, lastS);
        }
        if (c.tail_expr) {
            out << nextPrefix << "└── TailExpr\n";
            printExpr(c.tail_expr.get(), nextPrefix + "    ", true);
        }
    }
};

// ─────────────────────────────────────────────
//  Test runner
// ─────────────────────────────────────────────

struct TestCase
{
    std::string name;
    std::string code;
    bool        expect_fail;
};

static int passed = 0;
static int failed = 0;
static std::vector<std::string> failed_test_names;

static void runTest(const TestCase& tc, std::ostream& out)
{
    out << "\n── " << tc.name << " ──\n";
    out << tc.code << "\n";
    out << std::string(50, '-') << "\n";

    try
    {
        Lexer lexer(tc.code);
        Parser parser(lexer);
        Program program = parser.parse();

        if (tc.expect_fail)
        {
            out << "FAIL — expected parse error but none thrown\n";
            failed++;
            failed_test_names.push_back(tc.name);
            return;
        }

        // Note: For the file output, you might want to redirect 
        // std::cout or modify printStmt to take an ostream.
        // For now, this tracks the logic:
        ASTPrinter printer(out);
        printer.print(program);
        out << "PASS\n";
        passed++;
    }
    catch (const ParseError& e)
    {
        if (tc.expect_fail)
        {
            out << "PASS (expected error: " << e.what() << ")\n";
            passed++;
        }
        else
        {
            out << "FAIL — ParseError: " << e.what() << "\n";
            failed++;
            failed_test_names.push_back(tc.name);
        }
    }
}

// ─────────────────────────────────────────────
//  Test cases
// ─────────────────────────────────────────────

int main()
{
    std::vector<TestCase> tests = {

        // ════════════════════════════════════════════════════════
        //  SECTION 1 — OPERATOR PRECEDENCE & ASSOCIATIVITY
        // ════════════════════════════════════════════════════════

        {
            "Prec: * before +",
            "let x = 1 + 2 * 3 + 4",
            false
            // expect: +( +(1, *(2,3)), 4 )
        },
        {
            "Prec: unary before binary",
            "let x = -1 + -2 * -3",
            false
            // expect: +(-(1), *(-(2), -(3)))
        },
        {
            "Prec: @ above * /",
            "let x = 2 * a @ b + 1",
            false
            // expect: +( *(2, @(a,b)), 1 )  — @ binds tighter than *
        },
        {
            "Prec: comparison below arithmetic",
            "let ok = a + 1 > b - 2",
            false
            // expect: >( +(a,1), -(b,2) )
        },
        {
            "Prec: && above ||",
            "let x = a || b && c || d",
            false
            // expect: ||( ||(a, &&(b,c)), d )
        },
        {
            "Prec: == below comparison",
            "let x = a < b == c > d",
            false
            // expect: ==( <(a,b), >(c,d) )
        },
        {
            "Prec: pipe is lowest",
            "let x = a + b |> f |> g",
            false
            // expect: |>( |>(+(a,b), f), g )
        },
        {
            "Assoc: right-assoc assign",
            "a = b = c = 0",
            false
            // expect: =(a, =(b, =(c, 0)))
        },
        {
            "Assoc: left-assoc subtract",
            "let x = 10 - 3 - 2 - 1",
            false
            // expect: -(-(-(10,3),2),1)
        },
        {
            "Prec: grouped overrides",
            "let x = (1 + 2) * (3 + 4)",
            false
        },
        {
            "Prec: deeply grouped",
            "let x = ((((a + b))))",
            false
        },
        {
            "Prec: unary chain !!",
            "let x = !!false",
            false
        },
        {
            "Prec: unary chain ---",
            "let x = ---42",
            false
        },

        // ════════════════════════════════════════════════════════
        //  SECTION 2 — POSTFIX CHAINING
        // ════════════════════════════════════════════════════════

        {
            "Postfix: deep field chain",
            "let x = a.b.c.d.e",
            false
        },
        {
            "Postfix: call result indexed",
            "let x = get_layer(0)[2]",
            false
        },
        {
            "Postfix: scope then call",
            "let x = torch::nn::Linear(128, 64)",
            false
        },
        {
            "Postfix: field then call",
            "let x = model.forward(input)",
            false
        },
        {
            "Postfix: index then field",
            "let x = layers[i].weights",
            false
        },
        {
            "Postfix: index then call",
            "let x = ops[0](x, y)",
            false
        },
        {
            "Postfix: call chain",
            "let x = a()()()",
            false
        },
        {
            "Postfix: multi-arg call with expressions",
            "let x = loss(pred * scale, target + offset)",
            false
        },
        {
            "Postfix: scope on result of field access",
            "let x = config.mode::default",
            false
        },
        {
            "Postfix: channel send on complex lhs",
            "workers[0] <- process(batch)",
            false
        },

        // ════════════════════════════════════════════════════════
        //  SECTION 3 — COMPLEX FUNCTION DECLARATIONS
        // ════════════════════════════════════════════════════════

        {
            "Fn: no params, explicit void",
            "fn noop() -> void { }",
            false
        },
        {
            "Fn: many params",
            "fn f(a: i32, b: i32, c: f32, d: f64, e: bool, g: str) -> void { }",
            false
        },
        {
            "Fn: returns Tensor",
            "fn make_eye(n: i32) -> Tensor { Tensor#(f32, [3,3])[1.0,0.0,0.0; 0.0,1.0,0.0; 0.0,0.0,1.0] }",
            false
        },
        {
            "Fn: multi-generic",
            "fn zip#(A, B, C)(a: A, b: B) -> C { a }",
            false
        },
        {
            "Fn: body with mixed stmts and tail",
            R"(
fn relu(x: f32) -> f32 {
    let neg = x < 0.0
    if neg { 0.0 } else { x }
}
            )",
            false
        },
        {
            "Fn: nested fn declarations",
            R"(
fn outer(x: f32) -> f32 {
    fn inner(y: f32) -> f32 { y * 2.0 }
    inner(x) + 1.0
}
            )",
            false
        },
        {
            "Fn: calls itself (recursion syntax check only)",
            R"(
fn fib(n: i32) -> i32 {
    if n <= 1 { n } else { fib(n - 1) + fib(n - 2) }
}
            )",
            false
        },
        {
            "Fn: lambda stored in let, called immediately",
            "let result = fn(x: f32) -> f32 { x * x }(3.0)",
            false
        },
        {
            "Fn: lambda passed as argument",
            "let y = apply(fn(x: f32) -> f32 { x + 1.0 }, 5.0)",
            false
        },
        {
            "Fn: generic lambda",
            "let id = fn#(T)(x: T) -> T { x }",
            false
        },

        // ════════════════════════════════════════════════════════
        //  SECTION 4 — CONTROL FLOW COMPLEXITY
        // ════════════════════════════════════════════════════════

        {
            "If: condition is a call",
            "if is_valid(x) { process(x) }",
            false
        },
        {
            "If: condition is complex expression",
            "if a > 0 && b < 10 || c == d { ok() }",
            false
        },
        {
            "If: deeply nested else-if chain",
            R"(
if x == 0 { let r = "zero"
} else if x == 1 { let r = "one"
} else if x == 2 { let r = "two"
} else if x == 3 { let r = "three"
} else { let r = "many" }
            )",
            false
        },
        {
            "If: both branches are expressions with math",
            R"(
fn clamp(x: f32, lo: f32, hi: f32) -> f32 {
    if x < lo { lo } else if x > hi { hi } else { x }
}
            )",
            false
        },
        {
            "While: complex condition with && and call",
            R"(
while !done() && steps < max_steps {
    step += 1
}
            )",
            false
        },
        {
            "While: body has nested if",
            R"(
while epoch < 100 {
    if loss < 0.01 {
        break
    }
    epoch += 1
}
            )",
            false
        },
        {
            "Match: mixed expr and stmt arms",
            R"(
match status {
    0 => "ok"
    1 => let msg = "warn"
    2 => {
        let msg = "error"
        log(msg)
        msg
    }
    _ => "unknown"
}
            )",
            false
        },
        {
            "Match: pattern is a field access",
            R"(
match shape.kind {
    Circle  => area * 3.14
    Square  => area
    _ => 0.0
}
            )",
            false
        },
        {
            "Match: guard with complex condition",
            R"(
fn bucket(n: i32) -> str {
    match n {
        x if x < 0     => "negative"
        x if x == 0    => "zero"
        x if x < 10    => "small"
        x if x < 100   => "medium"
        _               => "large"
    }
}
            )",
            false
        },
        {
            "Match: nested match inside arm",
            R"(
match outer {
    0 => match inner {
        0 => "both zero"
        _ => "inner nonzero"
    }
    _ => "outer nonzero"
}
            )",
            false
        },

        // ════════════════════════════════════════════════════════
        //  SECTION 5 — BLOCK & IMPLICIT RETURN
        // ════════════════════════════════════════════════════════

        {
            "Block: stmts only, no tail",
            "{ let a = 1\n let b = 2 }",
            false
        },
        {
            "Block: single expression tail",
            "{ 42 }",
            false
        },
        {
            "Block: complex tail expression",
            "{ let x = 3\n let y = 4\n x * x + y * y }",
            false
        },
        {
            "Block: nested blocks",
            "{\n"
            "    let a = { let x = 1\n x + 1 }\n"
            "    let b = { let y = 2\n y * 2 }\n"
            "    a + b\n"
            "}",
            false
        },
        {
            "Implicit return: last stmt is if-expr",
            R"(
fn abs(x: f32) -> f32 {
    let neg = x < 0.0
    if neg { -x } else { x }
}
            )",
            false
        },
        {
            "Implicit return: last stmt is match-expr",
            R"(
fn sign(n: i32) -> i32 {
    match n {
        x if x > 0 => 1
        x if x < 0 => -1
        _          => 0
    }
}
            )",
            false
        },
        {
            "Implicit return: last stmt is block-expr",
            R"(
fn compute(x: f32) -> f32 {
    let t = x * 2.0
    { t + t }
}
            )",
            false
        },

        // ════════════════════════════════════════════════════════
        //  SECTION 6 — TENSOR & ML EXPRESSIONS
        // ════════════════════════════════════════════════════════

        {
            "Tensor: 1D row vector",
            "let v = Tensor#(f32, [1,4])[1.0, 2.0, 3.0, 4.0]",
            false
        },
        {
            "Tensor: 3x3 identity",
            "let I = Tensor#(f32, [3,3])[1.0,0.0,0.0; 0.0,1.0,0.0; 0.0,0.0,1.0]",
            false
        },
        {
            "Tensor: matrix multiply chain",
            "let out = W2 @ relu(W1 @ x + b1) + b2",
            false
        },
        {
            "Tensor: element-wise ops after matmul",
            "let x = (A @ B + C) * mask",
            false
        },
        {
            "Grad: chained over multiple params",
            R"(
fn update(model: Tensor, loss: f32) -> Tensor {
    let g = grad(loss, model)
    model - g * 0.001
}
            )",
            false
        },
        {
            "Full: transformer attention forward pass",
            R"(
fn attention(Q: Tensor, K: Tensor, V: Tensor, scale: f32) -> Tensor {
    let scores = Q @ K + scale
    let weights = softmax(scores)
    weights @ V
}
            )",
            false
        },
        {
            "Full: two-layer MLP",
            R"(
fn mlp(x: Tensor, W1: Tensor, b1: Tensor, W2: Tensor, b2: Tensor) -> Tensor {
    let h = relu(W1 @ x + b1)
    W2 @ h + b2
}
            )",
            false
        },
        {
            "Full: training loop with pipeline",
            R"(
fn train(model: Tensor, dataset: Queue) -> f32 {
    let total_loss = 0.0
    while !empty(dataset) {
        let batch = pop(dataset)
        let loss  = batch |> preprocess |> forward(model) |> compute_loss
        let grads = grad(loss, model)
        model -= grads * 0.001
        total_loss += loss
    }
    total_loss
}
            )",
            false
        },

        // ════════════════════════════════════════════════════════
        //  SECTION 7 — ASYNC / DISTRIBUTED
        // ════════════════════════════════════════════════════════

        {
            "Async: await in let binding",
            R"(
fn load_data() -> Tensor {
    let raw = await fetch("dataset.bin")
    let t   = parse_tensor(raw)
    t
}
            )",
            false
        },
        {
            "Async: await result used in expression",
            "let loss = compute(await get_batch()) * scale",
            false
        },
        {
            "Async: multiple awaits in body",
            R"(
fn pipeline() -> f32 {
    let a = await step_one()
    let b = await step_two(a)
    let c = await step_three(b)
    c
}
            )",
            false
        },
        {
            "Distributed: spawn with channel comms",
            R"(
fn parallel_train(batches: Queue) -> void {
    spawn fn worker_a() {
        let x = await pop(batches)
        ch_a <- forward(x)
    }
    spawn fn worker_b() {
        let grad_a = await recv(ch_a)
        ch_b <- backward(grad_a)
    }
    let result = await recv(ch_b)
}
            )",
            false
        },
        {
            "Distributed: channel send inside while",
            R"(
fn producer(ch: Queue, n: i32) -> void {
    let i = 0
    while i < n {
        ch <- generate(i)
        i += 1
    }
}
            )",
            false
        },

        // ════════════════════════════════════════════════════════
        //  SECTION 8 — COLLECTIONS IN DEPTH
        // ════════════════════════════════════════════════════════

        {
            "Map: string keys and mixed value types",
            R"(let cfg = Map{"lr": 0.001, "momentum": 0.9, "epochs": 100, "decay": 0.0001})",
            false
        },
        {
            "Map: value is expression",
            R"(let m = Map{"a": x + y, "b": foo(z) * 2})",
            false
        },
        {
            "Set: identifiers as elements",
            "let seen = Set[a, b, c, d, e]",
            false
        },
        {
            "Tuple: heterogeneous by intent",
            "let stats = Tuple[mean, variance, std_dev, min_val, max_val]",
            false
        },
        {
            "Array: nested array literals",
            "let mat = [[1,2,3],[4,5,6],[7,8,9]]",
            false
        },
        {
            "Array: expressions as elements",
            "let v = [a+b, c*d, f(x), -y]",
            false
        },
        {
            "Queue: with function call elements",
            "let q = Queue[load(\"a\"), load(\"b\"), load(\"c\")]",
            false
        },

        // ════════════════════════════════════════════════════════
        //  SECTION 9 — REAL PROGRAM STRUCTURES
        // ════════════════════════════════════════════════════════

        {
            "Program: import then fn",
            R"(
import "std/math" as math
import "std/io"

fn main() -> void {
    let x = math::sqrt(2.0)
    print(x)
}
            )",
            false
        },
        {
            "Program: struct definition",
            R"(
                struct Point {
                    x: i32,
                    y: i32,
                    label: str
                }
            )",
            false
        },
        {
            "Program: Struct Instance Declaration",
            R"(
                struct Player { id: i32 }
                
                fn main() -> void {
                    let p1: Player = get_player()
                }
            )",
            false 
        },
        {
            "Program: Nested Struct Field Access",
            R"(
                struct Transform {
                    position: Vec3,
                    rotation: f32
                }

                fn update() -> void {
                    let t: Transform = get_transform()
                    let x_pos = t.position.x
                    t.rotation = 90.0
                }
            )",
            false
        },
        {
            "Program: multiple fns calling each other",
            R"(
fn square(x: f32) -> f32 { x * x }
fn cube(x: f32)   -> f32 { x * square(x) }
fn poly(x: f32)   -> f32 { cube(x) + square(x) + x + 1.0 }
            )",
            false
        },
        {
            "Program: pipeline with all stages",
            R"(
import "nn"
import "optim"

fn train_epoch(model: Tensor, data: Queue, lr: f32) -> f32 {
    let total = 0.0
    while !empty(data) {
        let batch  = pop(data)
        let pred   = batch |> augment |> model
        let loss   = cross_entropy(pred, batch.labels)
        let grads  = grad(loss, model)
        model     -= grads * lr
        total     += loss
    }
    total / len(data)
}

fn main() -> void {
    let model = nn::Sequential(
        nn::Linear(784, 256),
        nn::ReLU(),
        nn::Linear(256, 10)
    )
    let data  = load_mnist()
    let epoch = 0
    while epoch < 50 {
        let loss = train_epoch(model, data, 0.001)
        epoch   += 1
    }
}
            )",
            false
        },

        // ════════════════════════════════════════════════════════
        //  SECTION 10 — ERROR CASES
        // ════════════════════════════════════════════════════════

        {
            "ERROR: missing identifier after let",
            "let = 42",
            true
        },
        {
            "ERROR: missing colon in type annotation",
            "let x i32 = 42",
            true
        },
        {
            "ERROR: missing ) in fn params",
            "fn f(x: i32, y: i32 { }",
            true
        },
        {
            "ERROR: missing { in fn body",
            "fn f() -> i32 return 1",
            true
        },
        {
            "ERROR: missing } in fn body",
            "fn f() -> i32 { 1 + 2",
            true
        },
        {
            "ERROR: if without condition",
            "if { 1 }",
            true
        },
        {
            "ERROR: else without if",
            "else { 1 }",
            true
        },
        {
            "ERROR: while body missing {",
            "while x > 0 x -= 1",
            true
        },
        {
            "ERROR: match without subject",
            "match { 0 => 1 }",
            true
        },
        {
            "ERROR: match arm missing =>",
            "match x { 0 \"zero\" }",
            true
        },
        {
            "ERROR: unmatched bracket in array",
            "let x = [1, 2, 3",
            true
        },
        {
            "ERROR: grad missing comma",
            "let g = grad(loss weights)",
            true
        },
        {
            "ERROR: grad missing )",
            "let g = grad(loss, weights",
            true
        },
        {
            "ERROR: await missing expression",
            "let x = await",
            true
        },
        {
            "ERROR: channel send missing value",
            "ch <-",
            true
        },
        {
            "ERROR: fn missing name",
            "fn (x: i32) -> i32 { x }",
            true
        },
        {
            "ERROR: import missing path string",
            "import nn",
            true
        },
        {
            "ERROR: spawn without fn",
            "spawn { }",
            true
        },
        {
            "ERROR: empty generic params #()",
            "fn f#()(x: i32) -> i32 { x }",
            false  // empty generics is parse-valid, semantics issue later
        },
        {
            "ERROR: double assign operator",
            "let x == 1",
            true
        },
    };

    std::ofstream outFile("results.txt");
    if (!outFile.is_open()) {
        std::cerr << "Error: Could not create results.txt\n";
        return 1;
    }

    for (const auto& tc : tests)
    {
        // Run and print to both console and file
        runTest(tc, outFile); 
        // Optional: runTest(tc, std::cout); 
    }

    // Write Final Summary to file
    outFile << "\n" << std::string(50, '=') << "\n";
    outFile << "FINAL TEST SUMMARY\n";
    outFile << "Passed: " << passed << "\n";
    outFile << "Failed: " << failed << "\n";
    outFile << std::string(50, '=') << "\n";

    if (!failed_test_names.empty()) {
        outFile << "\nFAILED CASES:\n";
        for (const auto& name : failed_test_names) {
            outFile << " - " << name << "\n";
        }
    }

    outFile.close();
    std::cout << "Tests complete. Results written to results.txt\n";

    return 0;
}