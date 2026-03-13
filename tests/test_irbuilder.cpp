#include <gtest/gtest.h>
#include "../compiler/ir/IRBuilder.h"
#include "../compiler/ir/IRPrinter.h"
 
using namespace ir;
 
// ── Helpers to build stub AST nodes quickly ───────────────────────────────────
 
static std::unique_ptr<LitIntExpr> lit_int(int64_t v, TypePtr ty = nullptr)
{
    auto e = std::make_unique<LitIntExpr>();
    e->value = v;
    e->resolved_type = ty ? ty : Type::i32();
    return e;
}
 
static std::unique_ptr<LitFloatExpr> lit_float(double v, TypePtr ty = nullptr)
{
    auto e = std::make_unique<LitFloatExpr>();
    e->value = v;
    e->resolved_type = ty ? ty : Type::f32();
    return e;
}
 
static std::unique_ptr<LitBoolExpr> lit_bool(bool v)
{
    auto e = std::make_unique<LitBoolExpr>();
    e->value = v;
    e->resolved_type = Type::bool_();
    return e;
}
 
static std::unique_ptr<LitStrExpr> lit_str(const std::string& s)
{
    auto e = std::make_unique<LitStrExpr>();
    e->value = s;
    e->resolved_type = Type::str_();
    return e;
}
 
static std::unique_ptr<IdentExpr> ident(const std::string& name, TypePtr ty)
{
    auto e = std::make_unique<IdentExpr>();
    e->name = name;
    e->resolved_type = ty;
    return e;
}
 
static std::unique_ptr<BinExpr> binexp(std::string op,
                                        std::unique_ptr<ASTNode> lhs,
                                        std::unique_ptr<ASTNode> rhs,
                                        TypePtr ty)
{
    auto e = std::make_unique<BinExpr>();
    e->op  = std::move(op);
    e->lhs = std::move(lhs);
    e->rhs = std::move(rhs);
    e->resolved_type = ty;
    return e;
}
 
static std::unique_ptr<UnExpr> unexpr(std::string op,
                                       std::unique_ptr<ASTNode> operand,
                                       TypePtr ty)
{
    auto e = std::make_unique<UnExpr>();
    e->op      = std::move(op);
    e->operand = std::move(operand);
    e->resolved_type = ty;
    return e;
}
 
// ── IRBuilder fixture with a live module/function/block ───────────────────────
 
class IRBuilderTest : public ::testing::Test
{
protected:
    IRBuilder  builder;
    std::unique_ptr<IRModule> mod;
    Function*   fn  = nullptr;
    BasicBlock* bb  = nullptr;
 
    void SetUp() override
    {
        mod = std::make_unique<IRModule>("test.tcc");
        fn  = mod->add_function("@test", Type::fn({}, Type::void_()));
        bb  = fn->create_entry();
        builder.set_function(fn);
        builder.set_block(bb);
        builder.push_scope();
    }
};
 
// ─── 1. SSA name generation ───────────────────────────────────────────────────
 
TEST_F(IRBuilderTest, FreshNamesAreUnique)
{
    std::string a = builder.fresh();
    std::string b = builder.fresh();
    std::string c = builder.fresh();
    EXPECT_NE(a, b);
    EXPECT_NE(b, c);
    EXPECT_NE(a, c);
    // All start with %
    EXPECT_EQ(a[0], '%');
    EXPECT_EQ(b[0], '%');
}
 
TEST_F(IRBuilderTest, FreshWithHintFirstUseNoSuffix)
{
    std::string n = builder.fresh("result");
    EXPECT_EQ(n, "%result");
}
 
TEST_F(IRBuilderTest, FreshWithHintSecondUseHasSuffix)
{
    builder.fresh("x");          // %x
    std::string n2 = builder.fresh("x");  // %x1
    EXPECT_NE(n2, "%x");
    EXPECT_TRUE(n2.find("x") != std::string::npos);
}
 
// ─── 2. Scope management ─────────────────────────────────────────────────────
 
TEST_F(IRBuilderTest, DefineAndLookup)
{
    auto val = std::make_shared<Value>("%v", Type::f32());
    builder.define("v", val.get());
    EXPECT_EQ(builder.lookup("v"), val.get());
}
 
TEST_F(IRBuilderTest, LookupReturnsNullForUndefined)
{
    EXPECT_EQ(builder.lookup("nonexistent"), nullptr);
}
 
TEST_F(IRBuilderTest, ScopeShadowing)
{
    auto outer = std::make_shared<Value>("%outer", Type::f32());
    builder.define("x", outer.get());
 
    builder.push_scope();
    auto inner = std::make_shared<Value>("%inner", Type::f32());
    builder.define("x", inner.get());
 
    // Inner scope shadows outer
    EXPECT_EQ(builder.lookup("x"), inner.get());
 
    builder.pop_scope();
    // Back to outer
    EXPECT_EQ(builder.lookup("x"), outer.get());
}
 
TEST_F(IRBuilderTest, PopScopeRemovesDefinitions)
{
    builder.push_scope();
    auto val = std::make_shared<Value>("%tmp", Type::i32());
    builder.define("tmp", val.get());
    EXPECT_EQ(builder.lookup("tmp"), val.get());
    builder.pop_scope();
    EXPECT_EQ(builder.lookup("tmp"), nullptr);
}
 
// ─── 3. Literal lowering ──────────────────────────────────────────────────────
 
TEST_F(IRBuilderTest, LowerLitInt)
{
    auto node = lit_int(42, Type::i32());
    Value* v  = builder.lower_expr(*node);
    ASSERT_NE(v, nullptr);
    auto* c = dynamic_cast<ConstantInt*>(v);
    ASSERT_NE(c, nullptr);
    EXPECT_EQ(c->value, 42);
}
 
TEST_F(IRBuilderTest, LowerLitFloat)
{
    auto node = lit_float(3.14, Type::f32());
    Value* v  = builder.lower_expr(*node);
    auto* c   = dynamic_cast<ConstantFloat*>(v);
    ASSERT_NE(c, nullptr);
    EXPECT_DOUBLE_EQ(c->value, 3.14);
}
 
TEST_F(IRBuilderTest, LowerLitBool)
{
    auto node = lit_bool(true);
    Value* v  = builder.lower_expr(*node);
    auto* c   = dynamic_cast<ConstantBool*>(v);
    ASSERT_NE(c, nullptr);
    EXPECT_TRUE(c->value);
}
 
TEST_F(IRBuilderTest, LowerLitStr)
{
    auto node = lit_str("hello");
    Value* v  = builder.lower_expr(*node);
    auto* c   = dynamic_cast<ConstantString*>(v);
    ASSERT_NE(c, nullptr);
    EXPECT_EQ(c->value, "hello");
}
 
// ─── 4. Identifier lowering ───────────────────────────────────────────────────
 
TEST_F(IRBuilderTest, LowerIdentResolvesBoundValue)
{
    auto val = std::make_shared<Value>("%myval", Type::f32());
    builder.define("myval", val.get());
 
    auto node = ident("myval", Type::f32());
    Value* v  = builder.lower_expr(*node);
    EXPECT_EQ(v, val.get());
}
 
TEST_F(IRBuilderTest, LowerIdentUnresolvedThrows)
{
    auto node = ident("ghost", Type::f32());
    EXPECT_THROW(builder.lower_expr(*node), std::runtime_error);
}
 
TEST_F(IRBuilderTest, LowerMutableIdentEmitsLoad)
{
    // Mutable variable = alloca in scope
    auto* alloca = bb->emit<AllocaInst>("%x.slot", Type::f32());
    builder.define("x", alloca);
 
    auto node = ident("x", Type::f32());
    Value* v  = builder.lower_expr(*node);
    // Should produce a LoadInst
    auto* load = dynamic_cast<LoadInst*>(v);
    ASSERT_NE(load, nullptr);
    EXPECT_EQ(bb->insts.size(), 2u);  // alloca + load
}
 
// ─── 5. Binary / unary operator mapping ──────────────────────────────────────
 
TEST_F(IRBuilderTest, FloatAddEmitsFAdd)
{
    auto node = binexp("+",
        lit_float(1.0f, Type::f32()), lit_float(2.0f, Type::f32()),
        Type::f32());
    builder.lower_expr(*node);
 
    auto* binop = dynamic_cast<BinOpInst*>(bb->insts.back().get());
    ASSERT_NE(binop, nullptr);
    EXPECT_EQ(binop->op, BinOpCode::FAdd);
}
 
TEST_F(IRBuilderTest, IntSubEmitsSub)
{
    auto node = binexp("-",
        lit_int(5, Type::i32()), lit_int(3, Type::i32()),
        Type::i32());
    builder.lower_expr(*node);
 
    auto* binop = dynamic_cast<BinOpInst*>(bb->insts.back().get());
    ASSERT_NE(binop, nullptr);
    EXPECT_EQ(binop->op, BinOpCode::Sub);
}
 
TEST_F(IRBuilderTest, FloatNegEmitsFNeg)
{
    auto node = unexpr("-", lit_float(1.0, Type::f32()), Type::f32());
    builder.lower_expr(*node);
 
    auto* unop = dynamic_cast<UnOpInst*>(bb->insts.back().get());
    ASSERT_NE(unop, nullptr);
    EXPECT_EQ(unop->op, UnOpCode::FNeg);
}
 
TEST_F(IRBuilderTest, LogicalNotEmitsNot)
{
    auto node = unexpr("!", lit_bool(true), Type::bool_());
    builder.lower_expr(*node);
 
    auto* unop = dynamic_cast<UnOpInst*>(bb->insts.back().get());
    ASSERT_NE(unop, nullptr);
    EXPECT_EQ(unop->op, UnOpCode::Not);
}
 
// ─── 6. if-else CFG ──────────────────────────────────────────────────────────
 
TEST_F(IRBuilderTest, IfElseCreatesThreeBlocks)
{
    // Build: if (true) { } else { }
    auto stmt = std::make_unique<IfStmt>();
    stmt->resolved_type = Type::void_();
 
    auto cond_expr = lit_bool(true);
    stmt->cond = std::move(cond_expr);
    // Empty then/else bodies
    stmt->then_body.clear();
    stmt->else_body = std::make_unique<std::vector<std::unique_ptr<ASTNode>>>();
 
    size_t blocks_before = fn->blocks.size();  // 1 (entry)
    builder.lower_stmt(*stmt);
 
    // Should have added: if.true, if.false, if.merge = 3 new blocks
    EXPECT_EQ(fn->blocks.size(), blocks_before + 3);
}
 
TEST_F(IRBuilderTest, IfNoElseCreatesTwoNewBlocks)
{
    auto stmt = std::make_unique<IfStmt>();
    stmt->resolved_type = Type::void_();
    stmt->cond = lit_bool(false);
    stmt->then_body.clear();
    stmt->else_body = nullptr;
 
    size_t blocks_before = fn->blocks.size();
    builder.lower_stmt(*stmt);
 
    // Should have added: if.true, if.merge = 2 new blocks
    EXPECT_EQ(fn->blocks.size(), blocks_before + 2);
}
 
TEST_F(IRBuilderTest, IfEntryBlockTerminatedWithCondBr)
{
    auto stmt = std::make_unique<IfStmt>();
    stmt->resolved_type = Type::void_();
    stmt->cond = lit_bool(true);
    stmt->then_body.clear();
    stmt->else_body = nullptr;
 
    builder.lower_stmt(*stmt);
    // Entry block should be terminated (CondBranch was emitted)
    EXPECT_TRUE(bb->is_terminated());
}
 
// ─── 7. While loop CFG ───────────────────────────────────────────────────────
 
TEST_F(IRBuilderTest, WhileLoopCreatesThreeNewBlocks)
{
    auto stmt = std::make_unique<WhileStmt>();
    stmt->resolved_type = Type::void_();
    stmt->cond = lit_bool(false);   // immediately-false: body never entered
    stmt->body.clear();
 
    size_t before = fn->blocks.size();
    builder.lower_stmt(*stmt);
 
    // header + body + exit = 3 new blocks
    EXPECT_EQ(fn->blocks.size(), before + 3);
}
 
TEST_F(IRBuilderTest, WhileEntryFallsIntoHeader)
{
    auto stmt = std::make_unique<WhileStmt>();
    stmt->resolved_type = Type::void_();
    stmt->cond = lit_bool(false);
    stmt->body.clear();
 
    builder.lower_stmt(*stmt);
    // Entry block ends with an unconditional branch (to header)
    EXPECT_TRUE(bb->is_terminated());
    EXPECT_NE(dynamic_cast<BranchInst*>(bb->terminator()), nullptr);
}
 
// ─── 8. Let statement ────────────────────────────────────────────────────────
 
TEST_F(IRBuilderTest, ImmutableLetBindsNameDirectly)
{
    auto stmt = std::make_unique<LetStmt>();
    stmt->name       = "pi";
    stmt->is_mutable = false;
    stmt->init       = lit_float(3.14, Type::f32());
    stmt->resolved_type = Type::f32();
 
    builder.lower_stmt(*stmt);
 
    Value* v = builder.lookup("pi");
    ASSERT_NE(v, nullptr);
    // Should NOT be an alloca — just the constant
    EXPECT_EQ(dynamic_cast<AllocaInst*>(v), nullptr);
}
 
TEST_F(IRBuilderTest, MutableLetEmitsAllocaAndStore)
{
    auto stmt = std::make_unique<LetStmt>();
    stmt->name       = "count";
    stmt->is_mutable = true;
    stmt->init       = lit_int(0, Type::i32());
    stmt->resolved_type = Type::i32();
 
    builder.lower_stmt(*stmt);
 
    // Scope should point to the alloca
    Value* v = builder.lookup("count");
    ASSERT_NE(v, nullptr);
    EXPECT_NE(dynamic_cast<AllocaInst*>(v), nullptr);
 
    // Two instructions: alloca + store
    EXPECT_EQ(bb->insts.size(), 2u);
}
 
// ─── 9. Tensor call lowering ──────────────────────────────────────────────────
 
TEST_F(IRBuilderTest, TensorMatMulLowersToTensorOpInst)
{
    auto tf32 = Type::tensor(Type::f32());
 
    // Build the FieldExpr: ts::matmul
    auto field = std::make_unique<FieldExpr>();
    field->module_alias  = "ts";
    field->field         = "matmul";
    field->resolved_type = tf32;
    // (object is unused in lower_tensor_call)
    field->object = std::make_unique<IdentExpr>();
 
    // Build stub CallExpr
    auto call = std::make_unique<CallExpr>();
    call->resolved_type = tf32;
    call->callee = std::move(field);
 
    // Two tensor args already in scope
    auto x = std::make_shared<Value>("%x", tf32);
    auto w = std::make_shared<Value>("%w", tf32);
    builder.define("x", x.get());
    builder.define("w", w.get());
 
    auto arg_x = ident("x", tf32);
    auto arg_w = ident("w", tf32);
    call->args.push_back(std::move(arg_x));
    call->args.push_back(std::move(arg_w));
 
    Value* result = builder.lower_expr(*call);
    auto* tensor_op = dynamic_cast<TensorOpInst*>(result);
    ASSERT_NE(tensor_op, nullptr);
    EXPECT_EQ(tensor_op->op, TensorOpCode::MatMul);
    EXPECT_EQ(tensor_op->args.size(), 2u);
}
 
TEST_F(IRBuilderTest, TensorReluLowers)
{
    auto tf32 = Type::tensor(Type::f32());
 
    auto field = std::make_unique<FieldExpr>();
    field->module_alias  = "ts";
    field->field         = "relu";
    field->resolved_type = tf32;
    field->object        = std::make_unique<IdentExpr>();
 
    auto call = std::make_unique<CallExpr>();
    call->resolved_type = tf32;
    call->callee = std::move(field);
 
    auto x = std::make_shared<Value>("%x", tf32);
    builder.define("x", x.get());
    call->args.push_back(ident("x", tf32));
 
    Value* result = builder.lower_expr(*call);
    auto* op = dynamic_cast<TensorOpInst*>(result);
    ASSERT_NE(op, nullptr);
    EXPECT_EQ(op->op, TensorOpCode::Relu);
}
 
// ─── 10. Spawn / await lowering ──────────────────────────────────────────────
 
TEST_F(IRBuilderTest, SpawnExprEmitsSpawnInst)
{
    auto task = std::make_shared<Value>("%task_fn", Type::fn({}, Type::f32()));
    builder.define("task_fn", task.get());
 
    auto spawn = std::make_unique<SpawnExpr>();
    spawn->expr          = ident("task_fn", Type::fn({}, Type::f32()));
    spawn->resolved_type = Type::infer();
 
    Value* result = builder.lower_expr(*spawn);
    auto* si = dynamic_cast<SpawnInst*>(result);
    ASSERT_NE(si, nullptr);
    EXPECT_TRUE(result->name.find("h") != std::string::npos);
}
 
TEST_F(IRBuilderTest, AwaitExprEmitsAwaitInst)
{
    auto handle = std::make_shared<Value>("%h", Type::infer());
    builder.define("h", handle.get());
 
    auto await = std::make_unique<AwaitExpr>();
    await->expr          = ident("h", Type::infer());
    await->resolved_type = Type::f32();
 
    Value* result = builder.lower_expr(*await);
    auto* ai = dynamic_cast<AwaitInst*>(result);
    ASSERT_NE(ai, nullptr);
    EXPECT_EQ(ai->type->kind, Type::Kind::F32);
}
 
// ─── 11. Full function + printer round-trip ──────────────────────────────────
 
TEST(IRBuilderRoundTripTest, ScalarAddRoundTrip)
{
    // Build:
    //   fn @add(%a: f32, %b: f32) -> f32:
    //   entry:
    //     %sum = fadd %a, %b   ; f32
    //     ret %sum
 
    IRBuilder builder;
    auto mod = std::make_unique<IRModule>("add.tcc");
 
    auto* fn    = mod->add_function("@add", Type::fn({Type::f32(), Type::f32()}, Type::f32()));
    auto* a_arg = fn->add_param("%a", Type::f32());
    auto* b_arg = fn->add_param("%b", Type::f32());
    auto* entry = fn->create_entry();
    builder.set_function(fn);
    builder.set_block(entry);
    builder.push_scope();
    builder.define("a", a_arg);
    builder.define("b", b_arg);
 
    // Lower:  a + b
    auto add_expr = binexp("+", ident("a", Type::f32()), ident("b", Type::f32()), Type::f32());
    Value* sum = builder.lower_expr(*add_expr);
 
    // Lower: return sum
    auto ret = std::make_unique<ReturnStmt>();
    // Manually emit — ReturnStmt needs a value
    auto sumv = std::shared_ptr<Value>(sum, [](Value*){});
    entry->emit<ReturnInst>(sumv);
 
    std::string out = IRPrinter::print(*mod);
 
    EXPECT_TRUE(out.find("fn @add(") != std::string::npos);
    EXPECT_TRUE(out.find("fadd") != std::string::npos);
    EXPECT_TRUE(out.find("ret") != std::string::npos);
}
 
TEST(IRBuilderRoundTripTest, ForwardPassRoundTrip)
{
    // Build:
    //   fn @forward(%x: Tensor<f32>, %w: Tensor<f32>) -> Tensor<f32>:
    //   entry:
    //     %mm  = tensor.matmul %x, %w
    //     %out = tensor.relu %mm
    //     ret %out
 
    IRBuilder builder;
    auto mod  = std::make_unique<IRModule>("nn.tcc");
    auto tf32 = Type::tensor(Type::f32());
    auto* fn  = mod->add_function("@forward", Type::fn({tf32, tf32}, tf32));
    auto* x   = fn->add_param("%x", tf32);
    auto* w   = fn->add_param("%w", tf32);
    auto* bb  = fn->create_entry();
    builder.set_function(fn);
    builder.set_block(bb);
    builder.push_scope();
    builder.define("x", x);
    builder.define("w", w);
 
    // ts::matmul(x, w)
    auto mm_field  = std::make_unique<FieldExpr>();
    mm_field->field         = "matmul";
    mm_field->module_alias  = "ts";
    mm_field->resolved_type = tf32;
    mm_field->object        = std::make_unique<IdentExpr>();
 
    auto mm_call = std::make_unique<CallExpr>();
    mm_call->resolved_type = tf32;
    mm_call->callee = std::move(mm_field);
    mm_call->args.push_back(ident("x", tf32));
    mm_call->args.push_back(ident("w", tf32));
 
    Value* mm_val = builder.lower_expr(*mm_call);
    builder.define("mm", mm_val);
 
    // ts::relu(mm)
    auto relu_field = std::make_unique<FieldExpr>();
    relu_field->field         = "relu";
    relu_field->module_alias  = "ts";
    relu_field->resolved_type = tf32;
    relu_field->object        = std::make_unique<IdentExpr>();
 
    auto relu_call = std::make_unique<CallExpr>();
    relu_call->resolved_type = tf32;
    relu_call->callee = std::move(relu_field);
    relu_call->args.push_back(ident("mm", tf32));
 
    Value* out_val = builder.lower_expr(*relu_call);
    auto out_vp = std::shared_ptr<Value>(out_val, [](Value*){});
    bb->emit<ReturnInst>(out_vp);
 
    std::string text = IRPrinter::print(*mod);
 
    EXPECT_TRUE(text.find("tensor.matmul") != std::string::npos);
    EXPECT_TRUE(text.find("tensor.relu")   != std::string::npos);
    EXPECT_TRUE(text.find("Tensor<f32>")   != std::string::npos);
    EXPECT_TRUE(text.find("ret")           != std::string::npos);
}