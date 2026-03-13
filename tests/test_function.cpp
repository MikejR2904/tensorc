#include <gtest/gtest.h>
#include "../compiler/ir/IRModule.h"
 
using namespace ir;
 
static TypePtr make_fn_type(std::vector<TypePtr> params, TypePtr ret)
{
    return Type::fn(std::move(params), std::move(ret));
}
 
// ─── 1. Construction ─────────────────────────────────────────────────────────
 
TEST(FunctionTest, NameAndTypeStored)
{
    Function fn("@add", make_fn_type({Type::f32(), Type::f32()}, Type::f32()));
    EXPECT_EQ(fn.name, "@add");
    EXPECT_EQ(fn.type->kind, Type::Kind::Fn);
}
 
TEST(FunctionTest, DefaultFlags)
{
    Function fn("@f", make_fn_type({}, Type::void_()));
    EXPECT_FALSE(fn.is_async);
    EXPECT_FALSE(fn.is_exported);
    EXPECT_EQ(fn.parent_module, nullptr);
}
 
// ─── 2. add_param ────────────────────────────────────────────────────────────
 
TEST(FunctionTest, AddParamStoresIndexAndType)
{
    Function fn("@f", make_fn_type({Type::f32(), Type::i32()}, Type::void_()));
 
    auto* p0 = fn.add_param("%x", Type::f32());
    auto* p1 = fn.add_param("%n", Type::i32());
 
    EXPECT_EQ(fn.params.size(), 2u);
    EXPECT_EQ(p0->index, 0u);
    EXPECT_EQ(p1->index, 1u);
    EXPECT_EQ(p0->name,  "%x");
    EXPECT_EQ(p1->name,  "%n");
    EXPECT_EQ(p0->type->kind, Type::Kind::F32);
    EXPECT_EQ(p1->type->kind, Type::Kind::I32);
}
 
TEST(FunctionTest, ParamIsArgument)
{
    Function fn("@g", make_fn_type({Type::bool_()}, Type::void_()));
    auto* p = fn.add_param("%flag", Type::bool_());
    EXPECT_TRUE(p->is_argument());
}
 
// ─── 3. Block management ─────────────────────────────────────────────────────
 
TEST(FunctionTest, CreateEntryFirstBlock)
{
    Function fn("@h", make_fn_type({}, Type::void_()));
    auto* entry = fn.create_entry();
 
    ASSERT_NE(entry, nullptr);
    EXPECT_EQ(entry->label, "entry");
    EXPECT_EQ(fn.blocks.size(), 1u);
    EXPECT_EQ(fn.entry(), entry);
}
 
TEST(FunctionTest, AddBlockAppendsAfterEntry)
{
    Function fn("@h", make_fn_type({}, Type::void_()));
    fn.create_entry();
    auto* loop = fn.add_block("loop.header");
    auto* body = fn.add_block("loop.body");
    auto* exit = fn.add_block("exit");
 
    EXPECT_EQ(fn.blocks.size(), 4u);
    EXPECT_EQ(fn.blocks[1].get(), loop);
    EXPECT_EQ(fn.blocks[2].get(), body);
    EXPECT_EQ(fn.blocks[3].get(), exit);
}
 
TEST(FunctionTest, BlockParentSetToFunction)
{
    Function fn("@f", make_fn_type({}, Type::void_()));
    auto* entry = fn.create_entry();
    EXPECT_EQ(entry->parent, &fn);
}
 
// ─── 4. entry() on empty function asserts ────────────────────────────────────
 
TEST(FunctionTest, EntryOnEmptyFunctionAsserts)
{
    Function fn("@empty", make_fn_type({}, Type::void_()));
    EXPECT_DEATH(fn.entry(), ".*");
}
 
// ─── 5. find_value ───────────────────────────────────────────────────────────
 
TEST(FunctionTest, FindValueFindsParam)
{
    Function fn("@f", make_fn_type({Type::f32()}, Type::f32()));
    fn.add_param("%input", Type::f32());
 
    Value* found = fn.find_value("%input");
    ASSERT_NE(found, nullptr);
    EXPECT_EQ(found->name, "%input");
    EXPECT_TRUE(found->is_argument());
}
 
TEST(FunctionTest, FindValueFindsInstruction)
{
    Function fn("@f", make_fn_type({}, Type::f32()));
    auto* entry = fn.create_entry();
 
    auto v = std::make_shared<Value>("%slot", Type::f32());
    entry->emit<AllocaInst>("%alloc", Type::f32());
 
    // find_value walks bb->insts
    Value* found = fn.find_value("%alloc");
    ASSERT_NE(found, nullptr);
    EXPECT_EQ(found->name, "%alloc");
}
 
TEST(FunctionTest, FindValueReturnsNullForMissing)
{
    Function fn("@f", make_fn_type({}, Type::void_()));
    fn.create_entry();
    EXPECT_EQ(fn.find_value("%nonexistent"), nullptr);
}
 
// ─── 6. Flags ────────────────────────────────────────────────────────────────
 
TEST(FunctionTest, AsyncFlag)
{
    Function fn("@spawn_task", make_fn_type({}, Type::void_()));
    fn.is_async = true;
    EXPECT_TRUE(fn.is_async);
}
 
TEST(FunctionTest, ExportedFlag)
{
    Function fn("@public_api", make_fn_type({}, Type::f32()));
    fn.is_exported = true;
    EXPECT_TRUE(fn.is_exported);
}