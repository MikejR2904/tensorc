#include <gtest/gtest.h>
#include "../compiler/ir/IRPrinter.h"
 
using namespace ir;
 
// ── Helpers ───────────────────────────────────────────────────────────────────
 
static ValuePtr V(const std::string& name, TypePtr ty)
{
    return std::make_shared<Value>(name, ty);
}
 
static bool has(const std::string& text, const std::string& sub)
{
    return text.find(sub) != std::string::npos;
}
 
// ─── Module header ────────────────────────────────────────────────────────────
 
TEST(IRPrinterTest, ModuleHeaderPrinted)
{
    IRModule mod("src/main.tcc");
    std::string out = IRPrinter::print(mod);
    EXPECT_TRUE(has(out, "module \"src/main.tcc\""));
}
 
TEST(IRPrinterTest, ImportedModulePrinted)
{
    IRModule mod("main.tcc");
    IRModule utils("utils.tcc");
    mod.imports.push_back(&utils);
 
    std::string out = IRPrinter::print(mod);
    EXPECT_TRUE(has(out, "import \"utils.tcc\""));
}
 
// ─── Function signature ───────────────────────────────────────────────────────
 
TEST(IRPrinterTest, SyncFunctionSignature)
{
    IRModule mod("t.tcc");
    auto* fn = mod.add_function(
        "@add",
        Type::fn({Type::f32(), Type::f32()}, Type::f32()));
    fn->add_param("%a", Type::f32());
    fn->add_param("%b", Type::f32());
    fn->create_entry()->emit<ReturnInst>();
 
    std::string out = IRPrinter::print(mod);
    EXPECT_TRUE(has(out, "fn @add("));
    EXPECT_TRUE(has(out, "%a: f32"));
    EXPECT_TRUE(has(out, "%b: f32"));
    EXPECT_TRUE(has(out, "-> f32"));
}
 
TEST(IRPrinterTest, AsyncFunctionKeyword)
{
    IRModule mod("t.tcc");
    auto* fn = mod.add_function("@task",
        Type::fn({}, Type::void_()), /*is_async=*/true);
    fn->create_entry()->emit<ReturnInst>();
 
    std::string out = IRPrinter::print(mod);
    EXPECT_TRUE(has(out, "async fn @task"));
}
 
TEST(IRPrinterTest, BlockLabelPrinted)
{
    IRModule mod("t.tcc");
    auto* fn = mod.add_function("@f", Type::fn({}, Type::void_()));
    fn->create_entry()->emit<ReturnInst>();
 
    std::string out = IRPrinter::print(mod);
    EXPECT_TRUE(has(out, "entry:"));
}
 
// ─── Arithmetic instructions ──────────────────────────────────────────────────
 
TEST(IRPrinterTest, FAddInstruction)
{
    IRModule mod("t.tcc");
    auto* fn    = mod.add_function("@f", Type::fn({Type::f32(),Type::f32()}, Type::f32()));
    auto* a_arg = fn->add_param("%a", Type::f32());
    auto* b_arg = fn->add_param("%b", Type::f32());
    auto* entry = fn->create_entry();
 
    auto a_v = std::shared_ptr<Value>(a_arg, [](Value*){});
    auto b_v = std::shared_ptr<Value>(b_arg, [](Value*){});
    auto* sum = entry->emit<BinOpInst>("%sum", Type::f32(), BinOpCode::FAdd, a_v, b_v);
    auto sum_v = std::shared_ptr<Value>(sum, [](Value*){});
    entry->emit<ReturnInst>(sum_v);
 
    std::string out = IRPrinter::print(mod);
    EXPECT_TRUE(has(out, "%sum = fadd %a, %b"));
    EXPECT_TRUE(has(out, "; f32"));
    EXPECT_TRUE(has(out, "ret %sum"));
}
 
TEST(IRPrinterTest, IntAddInstruction)
{
    IRModule mod("t.tcc");
    auto* fn = mod.add_function("@f", Type::fn({Type::i32(), Type::i32()}, Type::i32()));
    auto* a  = fn->add_param("%a", Type::i32());
    auto* b  = fn->add_param("%b", Type::i32());
    auto* bb = fn->create_entry();
    auto av  = std::shared_ptr<Value>(a, [](Value*){});
    auto bv  = std::shared_ptr<Value>(b, [](Value*){});
    auto* r  = bb->emit<BinOpInst>("%r", Type::i32(), BinOpCode::Add, av, bv);
    auto rv  = std::shared_ptr<Value>(r, [](Value*){});
    bb->emit<ReturnInst>(rv);
 
    std::string out = IRPrinter::print(mod);
    EXPECT_TRUE(has(out, "%r = add %a, %b"));
}
 
TEST(IRPrinterTest, UnaryFNeg)
{
    IRModule mod("t.tcc");
    auto* fn = mod.add_function("@neg", Type::fn({Type::f32()}, Type::f32()));
    auto* x  = fn->add_param("%x", Type::f32());
    auto* bb = fn->create_entry();
    auto xv  = std::shared_ptr<Value>(x, [](Value*){});
    auto* r  = bb->emit<UnOpInst>("%neg", Type::f32(), UnOpCode::FNeg, xv);
    auto rv  = std::shared_ptr<Value>(r, [](Value*){});
    bb->emit<ReturnInst>(rv);
 
    std::string out = IRPrinter::print(mod);
    EXPECT_TRUE(has(out, "%neg = fneg %x"));
}
 
TEST(IRPrinterTest, CmpInstruction)
{
    IRModule mod("t.tcc");
    auto* fn = mod.add_function("@cmp", Type::fn({Type::f32(),Type::f32()}, Type::bool_()));
    auto* a  = fn->add_param("%a", Type::f32());
    auto* b  = fn->add_param("%b", Type::f32());
    auto* bb = fn->create_entry();
    auto av  = std::shared_ptr<Value>(a, [](Value*){});
    auto bv  = std::shared_ptr<Value>(b, [](Value*){});
    auto* r  = bb->emit<CmpInst>("%cond", CmpCode::Gt, av, bv);
    auto rv  = std::shared_ptr<Value>(r, [](Value*){});
    bb->emit<ReturnInst>(rv);
 
    std::string out = IRPrinter::print(mod);
    EXPECT_TRUE(has(out, "%cond = cmp.gt %a, %b"));
    EXPECT_TRUE(has(out, "; bool"));
}
 
// ─── Memory instructions ──────────────────────────────────────────────────────
 
TEST(IRPrinterTest, AllocaAndStore)
{
    IRModule mod("t.tcc");
    auto* fn    = mod.add_function("@f", Type::fn({Type::f32()}, Type::void_()));
    auto* x_arg = fn->add_param("%x", Type::f32());
    auto* bb    = fn->create_entry();
    auto  xv    = std::shared_ptr<Value>(x_arg, [](Value*){});
    auto* slot  = bb->emit<AllocaInst>("%slot", Type::f32());
    auto  sv    = std::shared_ptr<Value>(slot, [](Value*){});
    bb->emit<StoreInst>(xv, sv);
    bb->emit<ReturnInst>();
 
    std::string out = IRPrinter::print(mod);
    EXPECT_TRUE(has(out, "%slot = alloca f32"));
    EXPECT_TRUE(has(out, "store %x -> %slot"));
    EXPECT_TRUE(has(out, "ret void"));
}
 
TEST(IRPrinterTest, LoadInstruction)
{
    IRModule mod("t.tcc");
    auto* fn   = mod.add_function("@f", Type::fn({}, Type::f32()));
    auto* bb   = fn->create_entry();
    auto* slot = bb->emit<AllocaInst>("%slot", Type::f32());
    auto  sv   = std::shared_ptr<Value>(slot, [](Value*){});
    auto* ld   = bb->emit<LoadInst>("%val", Type::f32(), sv);
    auto  ldv  = std::shared_ptr<Value>(ld, [](Value*){});
    bb->emit<ReturnInst>(ldv);
 
    std::string out = IRPrinter::print(mod);
    EXPECT_TRUE(has(out, "%val = load %slot"));
}
 
// ─── Control flow ─────────────────────────────────────────────────────────────
 
TEST(IRPrinterTest, BranchInstruction)
{
    IRModule mod("t.tcc");
    auto* fn   = mod.add_function("@f", Type::fn({}, Type::void_()));
    auto* bb   = fn->create_entry();
    auto* next = fn->add_block("next");
    bb->emit<BranchInst>(next);
    next->emit<ReturnInst>();
 
    std::string out = IRPrinter::print(mod);
    EXPECT_TRUE(has(out, "br %next"));
}
 
TEST(IRPrinterTest, CondBranchInstruction)
{
    IRModule mod("t.tcc");
    auto* fn  = mod.add_function("@f", Type::fn({Type::bool_()}, Type::void_()));
    auto* c   = fn->add_param("%cond", Type::bool_());
    auto* bb  = fn->create_entry();
    auto* tbb = fn->add_block("if.true");
    auto* fbb = fn->add_block("if.false");
    auto  cv  = std::shared_ptr<Value>(c, [](Value*){});
    bb->emit<CondBranchInst>(cv, tbb, fbb);
    tbb->emit<ReturnInst>();
    fbb->emit<ReturnInst>();
 
    std::string out = IRPrinter::print(mod);
    EXPECT_TRUE(has(out, "br %cond, %if.true, %if.false"));
}
 
TEST(IRPrinterTest, PhiNode)
{
    IRModule mod("t.tcc");
    auto* fn    = mod.add_function("@f", Type::fn({}, Type::i32()));
    auto* entry = fn->create_entry();
    auto* merge = fn->add_block("merge");
 
    auto v0 = std::make_shared<ConstantInt>(0, Type::i32());
    auto v1 = std::make_shared<ConstantInt>(1, Type::i32());
 
    auto* phi = merge->emit<PhiInst>("%x", Type::i32());
    phi->add_incoming(v0, entry);
    phi->add_incoming(v1, merge);
    auto phiv = std::shared_ptr<Value>(phi, [](Value*){});
    merge->emit<ReturnInst>(phiv);
    entry->emit<BranchInst>(merge);
 
    std::string out = IRPrinter::print(mod);
    EXPECT_TRUE(has(out, "%x = phi"));
    EXPECT_TRUE(has(out, "%entry"));
    EXPECT_TRUE(has(out, "%merge"));
}
 
// ─── Tensor ops ───────────────────────────────────────────────────────────────
 
TEST(IRPrinterTest, MatMulAndRelu)
{
    IRModule mod("nn.tcc");
    auto tf32 = Type::tensor(Type::f32());
    auto* fn  = mod.add_function("@forward", Type::fn({tf32, tf32}, tf32));
    auto* x   = fn->add_param("%x", tf32);
    auto* w   = fn->add_param("%w", tf32);
    auto* bb  = fn->create_entry();
    auto xv   = std::shared_ptr<Value>(x, [](Value*){});
    auto wv   = std::shared_ptr<Value>(w, [](Value*){});
    auto* mm  = bb->emit<TensorOpInst>("%mm", tf32, TensorOpCode::MatMul,
                    std::vector<ValuePtr>{xv, wv});
    auto mmv  = std::shared_ptr<Value>(mm, [](Value*){});
    auto* rl  = bb->emit<TensorOpInst>("%out", tf32, TensorOpCode::Relu,
                    std::vector<ValuePtr>{mmv});
    auto rlv  = std::shared_ptr<Value>(rl, [](Value*){});
    bb->emit<ReturnInst>(rlv);
 
    std::string out = IRPrinter::print(mod);
    EXPECT_TRUE(has(out, "%mm = tensor.matmul %x, %w"));
    EXPECT_TRUE(has(out, "%out = tensor.relu %mm"));
    EXPECT_TRUE(has(out, "; Tensor<f32>"));
}
 
TEST(IRPrinterTest, TensorOpWithShapeAnnotation)
{
    IRModule mod("t.tcc");
    auto tf32 = Type::tensor(Type::f32());
    auto* fn  = mod.add_function("@f", Type::fn({tf32}, tf32));
    auto* x   = fn->add_param("%x", tf32);
    auto* bb  = fn->create_entry();
    auto xv   = std::shared_ptr<Value>(x, [](Value*){});
    auto* r   = bb->emit<TensorOpInst>("%r", tf32, TensorOpCode::Flatten,
                    std::vector<ValuePtr>{xv});
    r->inferred_shape = {1024};
    auto rv   = std::shared_ptr<Value>(r, [](Value*){});
    bb->emit<ReturnInst>(rv);
 
    std::string out = IRPrinter::print(mod);
    EXPECT_TRUE(has(out, "{shape=[1024]}"));
}
 
TEST(IRPrinterTest, TensorOpWithGradFlag)
{
    IRModule mod("t.tcc");
    auto tf32 = Type::tensor(Type::f32());
    auto* fn  = mod.add_function("@f", Type::fn({tf32}, tf32));
    auto* x   = fn->add_param("%x", tf32);
    auto* bb  = fn->create_entry();
    auto xv   = std::shared_ptr<Value>(x, [](Value*){});
    auto* r   = bb->emit<TensorOpInst>("%r", tf32, TensorOpCode::Sigmoid,
                    std::vector<ValuePtr>{xv});
    r->requires_grad = true;
    auto rv   = std::shared_ptr<Value>(r, [](Value*){});
    bb->emit<ReturnInst>(rv);
 
    std::string out = IRPrinter::print(mod);
    EXPECT_TRUE(has(out, "#grad"));
}
 
// ─── Async instructions ───────────────────────────────────────────────────────
 
TEST(IRPrinterTest, SpawnAndAwait)
{
    IRModule mod("async.tcc");
    auto tf32  = Type::tensor(Type::f32());
    auto* fn   = mod.add_function("@f", Type::fn({tf32}, Type::f32()), true);
    auto* x    = fn->add_param("%x", tf32);
    auto* bb   = fn->create_entry();
    auto xv    = std::shared_ptr<Value>(x, [](Value*){});
    auto* sum  = bb->emit<TensorOpInst>("%s", Type::f32(), TensorOpCode::Sum,
                     std::vector<ValuePtr>{xv});
    auto sv    = std::shared_ptr<Value>(sum, [](Value*){});
    auto* h    = bb->emit<SpawnInst>("%h", Type::infer(), sv);
    auto hv    = std::shared_ptr<Value>(h, [](Value*){});
    auto* r    = bb->emit<AwaitInst>("%r", Type::f32(), hv);
    auto rv    = std::shared_ptr<Value>(r, [](Value*){});
    bb->emit<ReturnInst>(rv);
 
    std::string out = IRPrinter::print(mod);
    EXPECT_TRUE(has(out, "%h = spawn %s"));
    EXPECT_TRUE(has(out, "%r = await %h"));
}
 
TEST(IRPrinterTest, ParallelForAndBarrier)
{
    IRModule mod("par.tcc");
    auto* fn   = mod.add_function("@f", Type::fn({Type::i64()}, Type::void_()));
    auto* n    = fn->add_param("%n", Type::i64());
    auto* bb   = fn->create_entry();
    auto nv    = std::shared_ptr<Value>(n, [](Value*){});
    auto body  = std::make_shared<Value>("%body", Type::fn({Type::i64()}, Type::void_()));
    bb->emit<ParallelForInst>(nv, body);
    bb->emit<BarrierInst>();
    bb->emit<ReturnInst>();
 
    std::string out = IRPrinter::print(mod);
    EXPECT_TRUE(has(out, "parallel_for %n, %body"));
    EXPECT_TRUE(has(out, "barrier"));
}
 
// ─── Cast / Reshape ───────────────────────────────────────────────────────────
 
TEST(IRPrinterTest, CastInstruction)
{
    IRModule mod("t.tcc");
    auto* fn  = mod.add_function("@f", Type::fn({Type::i32()}, Type::f32()));
    auto* i   = fn->add_param("%i", Type::i32());
    auto* bb  = fn->create_entry();
    auto iv   = std::shared_ptr<Value>(i, [](Value*){});
    auto* r   = bb->emit<CastInst>("%f", Type::f32(), iv);
    auto rv   = std::shared_ptr<Value>(r, [](Value*){});
    bb->emit<ReturnInst>(rv);
 
    std::string out = IRPrinter::print(mod);
    EXPECT_TRUE(has(out, "%f = cast %i to f32"));
}
 
// ─── Global constants ─────────────────────────────────────────────────────────
 
TEST(IRPrinterTest, GlobalFloatConstant)
{
    IRModule mod("consts.tcc");
    auto pi = std::make_shared<ConstantFloat>(3.14159, Type::f32());
    pi->name = "pi";
    mod.add_global(pi);
 
    std::string out = IRPrinter::print(mod);
    EXPECT_TRUE(has(out, "@pi"));
    EXPECT_TRUE(has(out, "; f32"));
}
 
// ─── Multiple functions separated by blank lines ──────────────────────────────
 
TEST(IRPrinterTest, MultipleFunctionsSeparated)
{
    IRModule mod("multi.tcc");
    auto* f0 = mod.add_function("@f0", Type::fn({}, Type::void_()));
    f0->create_entry()->emit<ReturnInst>();
    auto* f1 = mod.add_function("@f1", Type::fn({}, Type::void_()));
    f1->create_entry()->emit<ReturnInst>();
 
    std::string out = IRPrinter::print(mod);
    EXPECT_TRUE(has(out, "fn @f0"));
    EXPECT_TRUE(has(out, "fn @f1"));
    // Blank line separates them
    EXPECT_TRUE(has(out, "\n\n"));
}
 
// ─── Call instruction ─────────────────────────────────────────────────────────
 
TEST(IRPrinterTest, CallInstruction)
{
    IRModule mod("t.tcc");
    auto* fn = mod.add_function("@caller", Type::fn({Type::f32()}, Type::f32()));
    auto* x  = fn->add_param("%x", Type::f32());
    auto* bb = fn->create_entry();
    auto xv  = std::shared_ptr<Value>(x, [](Value*){});
    auto callee = std::make_shared<Value>("@callee", Type::fn({Type::f32()}, Type::f32()));
    auto* r  = bb->emit<CallInst>("%r", Type::f32(), callee, std::vector<ValuePtr>{xv});
    auto rv  = std::shared_ptr<Value>(r, [](Value*){});
    bb->emit<ReturnInst>(rv);
 
    std::string out = IRPrinter::print(mod);
    EXPECT_TRUE(has(out, "%r = call @callee(%x)"));
}