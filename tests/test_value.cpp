/// test_value.cpp
/// Tests for ir/Value.h
///
/// Coverage
/// ────────
///   1. Value base construction (name, type)
///   2. is_constant / is_argument / is_instruction predicates
///   3. Use-def chain: add_use / has_uses
///   4. All Constant subclasses (Int, Float, Bool, String, Tensor)
///   5. Argument index tracking
 
#include <gtest/gtest.h>
#include "../compiler/ir/Value.h"
 
using namespace ir;
 
// ─── 1. Value base ────────────────────────────────────────────────────────────
 
TEST(ValueTest, ConstructionStoresNameAndType)
{
    auto type = Type::f32();
    auto val  = std::make_shared<Value>("%0", type);
 
    EXPECT_EQ(val->name, "%0");
    EXPECT_EQ(val->type->kind, Type::Kind::F32);
}
 
TEST(ValueTest, DefaultPredicatesAreFalse)
{
    auto val = std::make_shared<Value>("%x", Type::i64());
    EXPECT_FALSE(val->is_constant());
    EXPECT_FALSE(val->is_argument());
    EXPECT_FALSE(val->is_instruction());
}
 
// ─── 2. Use-def chain ─────────────────────────────────────────────────────────
 
TEST(ValueTest, NoUsesInitially)
{
    auto val = std::make_shared<Value>("%a", Type::bool_());
    EXPECT_FALSE(val->has_uses());
    EXPECT_EQ(val->uses.size(), 0u);
}
 
TEST(ValueTest, AddUseTracksUser)
{
    // We use a raw Instruction pointer as a placeholder user.
    // Instruction is not fully constructed here — we only test the pointer tracking.
    auto val = std::make_shared<Value>("%b", Type::f32());
 
    // Cast a dummy address as stand-in (safe: we never dereference it in add_use)
    Instruction* fake_user = reinterpret_cast<Instruction*>(0xDEAD);
    val->add_use(fake_user);
 
    EXPECT_TRUE(val->has_uses());
    EXPECT_EQ(val->uses.size(), 1u);
    EXPECT_EQ(val->uses[0], fake_user);
}
 
TEST(ValueTest, MultipleUsesAccumulate)
{
    auto val = std::make_shared<Value>("%c", Type::i32());
    for (int i = 1; i <= 5; ++i)
        val->add_use(reinterpret_cast<Instruction*>(static_cast<uintptr_t>(i)));
    EXPECT_EQ(val->uses.size(), 5u);
}
 
// ─── 3. Argument ─────────────────────────────────────────────────────────────
 
TEST(ArgumentTest, IsArgumentPredicateTrue)
{
    Argument arg("%x", Type::f32(), 0);
    EXPECT_TRUE(arg.is_argument());
    EXPECT_FALSE(arg.is_constant());
}
 
TEST(ArgumentTest, IndexStored)
{
    Argument a0("%a", Type::i32(), 0);
    Argument a1("%b", Type::i32(), 1);
    Argument a2("%c", Type::f32(), 2);
 
    EXPECT_EQ(a0.index, 0u);
    EXPECT_EQ(a1.index, 1u);
    EXPECT_EQ(a2.index, 2u);
}
 
TEST(ArgumentTest, TypePreserved)
{
    Argument arg("%tensor_in", Type::tensor(Type::f32()), 0);
    EXPECT_EQ(arg.type->kind, Type::Kind::Tensor);
    EXPECT_EQ(arg.type->inner_type()->kind, Type::Kind::F32);
}
 
// ─── 4. ConstantInt ───────────────────────────────────────────────────────────
 
TEST(ConstantIntTest, IsConstantPredicateTrue)
{
    ConstantInt c(42, Type::i32());
    EXPECT_TRUE(c.is_constant());
    EXPECT_FALSE(c.is_argument());
}
 
TEST(ConstantIntTest, ValueStored)
{
    ConstantInt c(-7, Type::i64());
    EXPECT_EQ(c.value, -7);
}
 
TEST(ConstantIntTest, NameIncludesValue)
{
    ConstantInt c(512, Type::i32());
    EXPECT_NE(c.name.find("512"), std::string::npos);
}
 
TEST(ConstantIntTest, TypePreserved)
{
    ConstantInt c(0, Type::i64());
    EXPECT_EQ(c.type->kind, Type::Kind::I64);
}
 
// ─── 5. ConstantFloat ────────────────────────────────────────────────────────
 
TEST(ConstantFloatTest, ValueStored)
{
    ConstantFloat c(3.14, Type::f32());
    EXPECT_DOUBLE_EQ(c.value, 3.14);
}
 
TEST(ConstantFloatTest, TypeIsFloat)
{
    ConstantFloat c(1.0, Type::f64());
    EXPECT_EQ(c.type->kind, Type::Kind::F64);
}
 
// ─── 6. ConstantBool ─────────────────────────────────────────────────────────
 
TEST(ConstantBoolTest, TrueNameAndValue)
{
    ConstantBool t(true);
    EXPECT_TRUE(t.value);
    EXPECT_EQ(t.name, "true");
    EXPECT_EQ(t.type->kind, Type::Kind::Bool);
}
 
TEST(ConstantBoolTest, FalseNameAndValue)
{
    ConstantBool f(false);
    EXPECT_FALSE(f.value);
    EXPECT_EQ(f.name, "false");
}
 
// ─── 7. ConstantString ───────────────────────────────────────────────────────
 
TEST(ConstantStringTest, ValueStored)
{
    ConstantString s("hello world");
    EXPECT_EQ(s.value, "hello world");
    EXPECT_EQ(s.type->kind, Type::Kind::Str);
}
 
TEST(ConstantStringTest, EmptyString)
{
    ConstantString s("");
    EXPECT_TRUE(s.value.empty());
}
 
// ─── 8. ConstantTensor ───────────────────────────────────────────────────────
 
TEST(ConstantTensorTest, DataAndShapeStored)
{
    std::vector<float>   data  = {1.f, 2.f, 3.f, 4.f};
    std::vector<int64_t> shape = {2, 2};
    ConstantTensor ct(data, shape, Type::tensor(Type::f32()));
 
    EXPECT_EQ(ct.data.size(),  4u);
    EXPECT_EQ(ct.shape.size(), 2u);
    EXPECT_FLOAT_EQ(ct.data[3], 4.f);
    EXPECT_EQ(ct.shape[0], 2);
    EXPECT_EQ(ct.shape[1], 2);
}
 
TEST(ConstantTensorTest, TypeIsTensor)
{
    ConstantTensor ct({}, {0}, Type::tensor(Type::f32()));
    EXPECT_EQ(ct.type->kind, Type::Kind::Tensor);
}