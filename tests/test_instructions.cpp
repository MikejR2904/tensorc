#include <gtest/gtest.h>
#include "../compiler/ir/IRModule.h"   // brings in Instruction.h + defines BasicBlock
 
using namespace ir;
 
// ── Helpers ───────────────────────────────────────────────────────────────────
 
static ValuePtr make_val(const std::string& name, TypePtr ty)
{
    return std::make_shared<Value>(name, ty);
}
 
// ─── 1. BinOpInst ─────────────────────────────────────────────────────────────
 
TEST(BinOpInstTest, Construction)
{
    auto lhs  = make_val("%a", Type::f32());
    auto rhs  = make_val("%b", Type::f32());
    BinOpInst inst("%c", Type::f32(), BinOpCode::FAdd, lhs, rhs);
 
    EXPECT_EQ(inst.name, "%c");
    EXPECT_EQ(inst.op,   BinOpCode::FAdd);
    EXPECT_EQ(inst.type->kind, Type::Kind::F32);
}
 
TEST(BinOpInstTest, UsesTracked)
{
    auto lhs = make_val("%x", Type::i32());
    auto rhs = make_val("%y", Type::i32());
    BinOpInst inst("%z", Type::i32(), BinOpCode::Add, lhs, rhs);
 
    EXPECT_TRUE(lhs->has_uses());
    EXPECT_TRUE(rhs->has_uses());
    EXPECT_EQ(lhs->uses[0], &inst);
    EXPECT_EQ(rhs->uses[0], &inst);
}
 
TEST(BinOpInstTest, IsInstruction)
{
    auto v = make_val("%a", Type::i32());
    BinOpInst inst("%b", Type::i32(), BinOpCode::Sub, v, v);
    EXPECT_TRUE(inst.is_instruction());
    EXPECT_FALSE(inst.is_constant());
}
 
TEST(BinOpInstTest, IntegerDivision)
{
    auto lhs = make_val("%n", Type::i64());
    auto rhs = make_val("%d", Type::i64());
    BinOpInst inst("%q", Type::i64(), BinOpCode::Div, lhs, rhs);
    EXPECT_EQ(inst.op, BinOpCode::Div);
}
 
// ─── 2. UnOpInst ─────────────────────────────────────────────────────────────
 
TEST(UnOpInstTest, NegPreservesType)
{
    auto operand = make_val("%v", Type::f32());
    UnOpInst inst("%neg", Type::f32(), UnOpCode::FNeg, operand);
 
    EXPECT_EQ(inst.op, UnOpCode::FNeg);
    EXPECT_TRUE(operand->has_uses());
}
 
TEST(UnOpInstTest, NotOnBool)
{
    auto cond = make_val("%cond", Type::bool_());
    UnOpInst inst("%ncond", Type::bool_(), UnOpCode::Not, cond);
    EXPECT_EQ(inst.type->kind, Type::Kind::Bool);
}
 
// ─── 3. CmpInst ──────────────────────────────────────────────────────────────
 
TEST(CmpInstTest, ResultIsAlwaysBool)
{
    auto lhs = make_val("%p", Type::f32());
    auto rhs = make_val("%q", Type::f32());
    CmpInst inst("%r", CmpCode::Gt, lhs, rhs);
 
    EXPECT_EQ(inst.type->kind, Type::Kind::Bool);
}
 
TEST(CmpInstTest, AllCmpCodesConstruct)
{
    auto a = make_val("%a", Type::i32());
    auto b = make_val("%b", Type::i32());
    for (auto code : { CmpCode::Eq, CmpCode::Ne, CmpCode::Lt,
                       CmpCode::Le, CmpCode::Gt, CmpCode::Ge })
    {
        CmpInst inst("%r", code, a, b);
        EXPECT_EQ(inst.cmp, code);
    }
}
 
// ─── 4. AllocaInst ───────────────────────────────────────────────────────────
 
TEST(AllocaInstTest, AllocTypeStored)
{
    AllocaInst inst("%ptr", Type::f32());
    EXPECT_EQ(inst.alloc_type->kind, Type::Kind::F32);
    EXPECT_EQ(inst.type->kind,       Type::Kind::F32);
}
 
TEST(AllocaInstTest, TensorAlloca)
{
    AllocaInst inst("%t", Type::tensor(Type::f32()));
    EXPECT_EQ(inst.alloc_type->kind, Type::Kind::Tensor);
}
 
// ─── 5. LoadInst ─────────────────────────────────────────────────────────────
 
TEST(LoadInstTest, PtrTracked)
{
    auto ptr = make_val("%ptr", Type::f32());
    LoadInst inst("%val", Type::f32(), ptr);
 
    EXPECT_EQ(inst.ptr, ptr);
    EXPECT_TRUE(ptr->has_uses());
}
 
// ─── 6. StoreInst ────────────────────────────────────────────────────────────
 
TEST(StoreInstTest, VoidResult)
{
    auto val = make_val("%v", Type::f32());
    auto ptr = make_val("%p", Type::f32());
    StoreInst inst(val, ptr);
 
    EXPECT_EQ(inst.type->kind, Type::Kind::Void);
    EXPECT_TRUE(val->has_uses());
    EXPECT_TRUE(ptr->has_uses());
}
 
// ─── 7. BranchInst ───────────────────────────────────────────────────────────
 
TEST(BranchInstTest, TargetStored)
{
    BasicBlock fake_bb("loop.body");
    BranchInst inst(&fake_bb);
    EXPECT_EQ(inst.target, &fake_bb);
    EXPECT_EQ(inst.type->kind, Type::Kind::Void);
}
 
// ─── 8. CondBranchInst ───────────────────────────────────────────────────────
 
TEST(CondBranchInstTest, CondAndTargetsStored)
{
    auto cond = make_val("%cond", Type::bool_());
    BasicBlock true_bb("if.true"), false_bb("if.false");
 
    CondBranchInst inst(cond, &true_bb, &false_bb);
 
    EXPECT_EQ(inst.cond,     cond);
    EXPECT_EQ(inst.true_bb,  &true_bb);
    EXPECT_EQ(inst.false_bb, &false_bb);
    EXPECT_TRUE(cond->has_uses());
}
 
// ─── 9. ReturnInst ───────────────────────────────────────────────────────────
 
TEST(ReturnInstTest, VoidReturn)
{
    ReturnInst inst;
    EXPECT_FALSE(inst.val.has_value());
}
 
TEST(ReturnInstTest, ValueReturn)
{
    auto retval = make_val("%result", Type::f32());
    ReturnInst inst(retval);
 
    EXPECT_TRUE(inst.val.has_value());
    EXPECT_EQ(*inst.val, retval);
    EXPECT_TRUE(retval->has_uses());
}
 
// ─── 10. CallInst ────────────────────────────────────────────────────────────
 
TEST(CallInstTest, CalleeAndArgsTracked)
{
    auto callee = make_val("@my_fn", Type::fn({Type::f32()}, Type::f32()));
    auto arg0   = make_val("%a",     Type::f32());
 
    CallInst inst("%r", Type::f32(), callee, {arg0});
 
    EXPECT_EQ(inst.callee, callee);
    EXPECT_EQ(inst.args.size(), 1u);
    EXPECT_TRUE(callee->has_uses());
    EXPECT_TRUE(arg0->has_uses());
}
 
TEST(CallInstTest, TailCallFlag)
{
    auto callee = make_val("@f", Type::fn({}, Type::void_()));
    CallInst inst("", Type::void_(), callee, {}, /*is_tail=*/true);
    EXPECT_TRUE(inst.is_tail);
}
 
TEST(CallInstTest, MultipleArgs)
{
    auto callee = make_val("@g", Type::fn({Type::i32(), Type::i32()}, Type::i32()));
    auto a = make_val("%a", Type::i32());
    auto b = make_val("%b", Type::i32());
 
    CallInst inst("%r", Type::i32(), callee, {a, b});
    EXPECT_EQ(inst.args.size(), 2u);
}
 
// ─── 11. PhiInst ─────────────────────────────────────────────────────────────
 
TEST(PhiInstTest, EmptyInitially)
{
    PhiInst inst("%phi", Type::i32());
    EXPECT_TRUE(inst.incoming.empty());
}
 
TEST(PhiInstTest, AddIncomingAccumulates)
{
    PhiInst inst("%phi", Type::i32());
    BasicBlock bb0("pred0"), bb1("pred1");
 
    auto v0 = make_val("%v0", Type::i32());
    auto v1 = make_val("%v1", Type::i32());
 
    inst.add_incoming(v0, &bb0);
    inst.add_incoming(v1, &bb1);
 
    EXPECT_EQ(inst.incoming.size(), 2u);
    EXPECT_EQ(inst.incoming[0].first,  v0);
    EXPECT_EQ(inst.incoming[0].second, &bb0);
    EXPECT_EQ(inst.incoming[1].first,  v1);
    EXPECT_EQ(inst.incoming[1].second, &bb1);
    EXPECT_TRUE(v0->has_uses());
    EXPECT_TRUE(v1->has_uses());
}
 
// ─── 12. TensorOpInst ────────────────────────────────────────────────────────
 
TEST(TensorOpInstTest, MatMulConstruction)
{
    auto t1 = make_val("%a", Type::tensor(Type::f32()));
    auto t2 = make_val("%b", Type::tensor(Type::f32()));
 
    TensorOpInst inst("%c", Type::tensor(Type::f32()),
                      TensorOpCode::MatMul, {t1, t2});
 
    EXPECT_EQ(inst.op, TensorOpCode::MatMul);
    EXPECT_EQ(inst.args.size(), 2u);
    EXPECT_TRUE(t1->has_uses());
    EXPECT_TRUE(t2->has_uses());
}
 
TEST(TensorOpInstTest, DefaultFlagsAndShape)
{
    auto t = make_val("%t", Type::tensor(Type::f32()));
    TensorOpInst inst("%r", Type::f32(), TensorOpCode::Sum, {t});
 
    EXPECT_FALSE(inst.requires_grad);
    EXPECT_FALSE(inst.inferred_shape.has_value());
}
 
TEST(TensorOpInstTest, ShapeAnnotation)
{
    auto t = make_val("%t", Type::tensor(Type::f32()));
    TensorOpInst inst("%r", Type::tensor(Type::f32()), TensorOpCode::Reshape, {t});
 
    inst.inferred_shape = {64, 512};
    ASSERT_TRUE(inst.inferred_shape.has_value());
    EXPECT_EQ((*inst.inferred_shape)[0], 64);
    EXPECT_EQ((*inst.inferred_shape)[1], 512);
}
 
TEST(TensorOpInstTest, RequiresGradFlag)
{
    auto t = make_val("%t", Type::tensor(Type::f32()));
    TensorOpInst inst("%r", Type::tensor(Type::f32()), TensorOpCode::Relu, {t});
    inst.requires_grad = true;
    EXPECT_TRUE(inst.requires_grad);
}
 
TEST(TensorOpInstTest, BackwardOpCode)
{
    auto t = make_val("%loss", Type::f32());
    TensorOpInst inst("", Type::void_(), TensorOpCode::Backward, {t});
    EXPECT_EQ(inst.op, TensorOpCode::Backward);
}
 
TEST(TensorOpInstTest, ActivationChain)
{
    // relu(matmul(x, w)) — two tensor ops, use-def chain intact
    auto x = make_val("%x", Type::tensor(Type::f32()));
    auto w = make_val("%w", Type::tensor(Type::f32()));
 
    auto mm = std::make_shared<TensorOpInst>(
        "%mm", Type::tensor(Type::f32()), TensorOpCode::MatMul,
        std::vector<ValuePtr>{x, w});
 
    TensorOpInst relu("%out", Type::tensor(Type::f32()),
                      TensorOpCode::Relu,
                      std::vector<ValuePtr>{mm});
 
    EXPECT_TRUE(x->has_uses());
    EXPECT_TRUE(w->has_uses());
    EXPECT_TRUE(mm->has_uses());
}
 
// ─── 13. SpawnInst ───────────────────────────────────────────────────────────
 
TEST(SpawnInstTest, TaskTracked)
{
    auto task = make_val("%fn", Type::fn({}, Type::void_()));
    SpawnInst inst("%handle", Type::infer(), task);
 
    EXPECT_EQ(inst.task, task);
    EXPECT_TRUE(task->has_uses());
}
 
// ─── 14. AwaitInst ───────────────────────────────────────────────────────────
 
TEST(AwaitInstTest, HandleTracked)
{
    auto handle = make_val("%h", Type::infer());
    AwaitInst inst("%result", Type::tensor(Type::f32()), handle);
 
    EXPECT_EQ(inst.handle, handle);
    EXPECT_TRUE(handle->has_uses());
    EXPECT_EQ(inst.type->kind, Type::Kind::Tensor);
}
 
// ─── 15. ParallelForInst ─────────────────────────────────────────────────────
 
TEST(ParallelForInstTest, NAndBodyTracked)
{
    auto n    = make_val("%n",    Type::i64());
    auto body = make_val("%body", Type::fn({Type::i64()}, Type::void_()));
 
    ParallelForInst inst(n, body);
 
    EXPECT_EQ(inst.n,       n);
    EXPECT_EQ(inst.body_fn, body);
    EXPECT_TRUE(n->has_uses());
    EXPECT_TRUE(body->has_uses());
    EXPECT_EQ(inst.type->kind, Type::Kind::Void);
}
 
// ─── 16. BarrierInst ─────────────────────────────────────────────────────────
 
TEST(BarrierInstTest, VoidResultNoOperands)
{
    BarrierInst inst;
    EXPECT_EQ(inst.type->kind, Type::Kind::Void);
    EXPECT_TRUE(inst.uses.empty());
}
 
// ─── 17. CastInst ────────────────────────────────────────────────────────────
 
TEST(CastInstTest, TargetTypeStored)
{
    auto src = make_val("%i", Type::i32());
    CastInst inst("%f", Type::f32(), src);
 
    EXPECT_EQ(inst.target_type->kind, Type::Kind::F32);
    EXPECT_TRUE(src->has_uses());
}
 
// ─── 18. ReshapeInst ─────────────────────────────────────────────────────────
 
TEST(ReshapeInstTest, TensorAndShapeTracked)
{
    auto tensor = make_val("%t",     Type::tensor(Type::f32()));
    auto shape  = make_val("%shape", Type::array(Type::i64()));
 
    ReshapeInst inst("%r", Type::tensor(Type::f32()), tensor, shape);
 
    EXPECT_EQ(inst.tensor, tensor);
    EXPECT_EQ(inst.shape,  shape);
    EXPECT_TRUE(tensor->has_uses());
    EXPECT_TRUE(shape->has_uses());
}