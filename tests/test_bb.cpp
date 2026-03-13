#include <gtest/gtest.h>
#include "../compiler/ir/IRModule.h"
 
using namespace ir;
 
static ValuePtr make_val(const std::string& name, TypePtr ty)
{
    return std::make_shared<Value>(name, ty);
}
 
// ─── 1. Construction ─────────────────────────────────────────────────────────
 
TEST(BasicBlockTest, LabelStored)
{
    BasicBlock bb("entry");
    EXPECT_EQ(bb.label, "entry");
}
 
TEST(BasicBlockTest, EmptyOnConstruction)
{
    BasicBlock bb("start");
    EXPECT_TRUE(bb.insts.empty());
    EXPECT_FALSE(bb.is_terminated());
    EXPECT_EQ(bb.terminator(), nullptr);
}
 
// ─── 2. emit<T>() ────────────────────────────────────────────────────────────
 
TEST(BasicBlockTest, EmitSetsParent)
{
    BasicBlock bb("entry");
    auto lhs = make_val("%a", Type::f32());
    auto rhs = make_val("%b", Type::f32());
 
    auto* inst = bb.emit<BinOpInst>("%c", Type::f32(), BinOpCode::FAdd, lhs, rhs);
 
    ASSERT_NE(inst, nullptr);
    EXPECT_EQ(inst->parent, &bb);
}
 
TEST(BasicBlockTest, EmitAppendsInOrder)
{
    BasicBlock bb("body");
    auto v = make_val("%v", Type::i32());
 
    auto* i0 = bb.emit<AllocaInst>("%slot0", Type::i32());
    auto* i1 = bb.emit<AllocaInst>("%slot1", Type::f32());
    auto* i2 = bb.emit<LoadInst>("%ld",      Type::i32(), v);
 
    ASSERT_EQ(bb.insts.size(), 3u);
    EXPECT_EQ(bb.insts[0].get(), i0);
    EXPECT_EQ(bb.insts[1].get(), i1);
    EXPECT_EQ(bb.insts[2].get(), i2);
}
 
// ─── 3. is_terminated() ──────────────────────────────────────────────────────
 
TEST(BasicBlockTest, NotTerminatedWithNonTerminator)
{
    BasicBlock bb("mid");
    auto v = make_val("%x", Type::f32());
    bb.emit<AllocaInst>("%slot", Type::f32());
    EXPECT_FALSE(bb.is_terminated());
}
 
TEST(BasicBlockTest, TerminatedAfterReturn)
{
    BasicBlock bb("exit");
    auto retval = make_val("%r", Type::f32());
    bb.emit<ReturnInst>(retval);
    EXPECT_TRUE(bb.is_terminated());
}
 
TEST(BasicBlockTest, TerminatedAfterVoidReturn)
{
    BasicBlock bb("exit");
    bb.emit<ReturnInst>();
    EXPECT_TRUE(bb.is_terminated());
}
 
TEST(BasicBlockTest, TerminatedAfterBranch)
{
    BasicBlock bb("body");
    BasicBlock target("next");
    bb.emit<BranchInst>(&target);
    EXPECT_TRUE(bb.is_terminated());
}
 
TEST(BasicBlockTest, TerminatedAfterCondBranch)
{
    BasicBlock bb("check");
    BasicBlock tbb("true"), fbb("false");
    auto cond = make_val("%c", Type::bool_());
    bb.emit<CondBranchInst>(cond, &tbb, &fbb);
    EXPECT_TRUE(bb.is_terminated());
}
 
// ─── 4. terminator() accessor ────────────────────────────────────────────────
 
TEST(BasicBlockTest, TerminatorReturnsLastInst)
{
    BasicBlock bb("loop.end");
    BasicBlock next("after");
 
    bb.emit<AllocaInst>("%tmp", Type::i32());
    auto* br = bb.emit<BranchInst>(&next);
 
    EXPECT_EQ(bb.terminator(), br);
}
 
// ─── 5. Predecessors / successors default empty ───────────────────────────────
 
TEST(BasicBlockTest, PredsSuccsEmptyByDefault)
{
    BasicBlock bb("block");
    EXPECT_TRUE(bb.preds.empty());
    EXPECT_TRUE(bb.succs.empty());
}
 
// ─── 6. Mixed instruction sequence ───────────────────────────────────────────
 
TEST(BasicBlockTest, FullStraightLineSequence)
{
    // Models:  %slot = alloca f32
    //          store 1.0 -> %slot
    //          %val = load %slot
    //          %doubled = %val * 2.0
    //          ret %doubled
    BasicBlock bb("straight");
 
    auto one  = std::make_shared<ConstantFloat>(1.0, Type::f32());
    auto two  = std::make_shared<ConstantFloat>(2.0, Type::f32());
 
    auto* alloca  = bb.emit<AllocaInst>("%slot",   Type::f32());
    auto alloca_v = std::shared_ptr<Value>(alloca, [](Value*){});  // non-owning alias
    // For test simplicity use make_val instead of alloca_v
    auto slot_val = make_val("%slot", Type::f32());
 
    bb.emit<StoreInst>(one, slot_val);
    auto* load    = bb.emit<LoadInst>("%val",    Type::f32(), slot_val);
    auto load_v   = std::shared_ptr<Value>(load, [](Value*){});
 
    auto load_val = make_val("%val", Type::f32());
    bb.emit<BinOpInst>("%doubled", Type::f32(), BinOpCode::FMul, load_val, two);
 
    auto doubled  = make_val("%doubled", Type::f32());
    bb.emit<ReturnInst>(doubled);
 
    EXPECT_EQ(bb.insts.size(), 5u);
    EXPECT_TRUE(bb.is_terminated());
}