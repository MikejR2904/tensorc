#include <gtest/gtest.h>
#include "../compiler/ir/IRModule.h"
 
using namespace ir;
 
static ValuePtr make_val(const std::string& name, TypePtr ty)
{
    return std::make_shared<Value>(name, ty);
}
 
// ─── 1. Scalar add function ──────────────────────────────────────────────────
//
//  fn add(a: f32, b: f32) -> f32 {
//      return a + b;
//  }
 
TEST(IntegrationTest, ScalarAddFunction)
{
    IRModule mod("scalar.tcc");
 
    auto fn_type = Type::fn({Type::f32(), Type::f32()}, Type::f32());
    auto* fn     = mod.add_function("@add", fn_type);
    auto* a      = fn->add_param("%a", Type::f32());
    auto* b      = fn->add_param("%b", Type::f32());
    auto* entry  = fn->create_entry();
 
    auto a_val = std::shared_ptr<Value>(a, [](Value*){});
    auto b_val = std::shared_ptr<Value>(b, [](Value*){});
 
    auto* sum = entry->emit<BinOpInst>("%sum", Type::f32(), BinOpCode::FAdd, a_val, b_val);
    auto sum_val = std::shared_ptr<Value>(sum, [](Value*){});
    entry->emit<ReturnInst>(sum_val);
 
    // Structure assertions
    EXPECT_EQ(fn->params.size(),      2u);
    EXPECT_EQ(entry->insts.size(),    2u);
    EXPECT_TRUE(entry->is_terminated());
    EXPECT_TRUE(a_val->has_uses());
    EXPECT_TRUE(b_val->has_uses());
}
 
// ─── 2. Tensor forward pass ──────────────────────────────────────────────────
//
//  fn forward(x: Tensor<f32>, w: Tensor<f32>) -> Tensor<f32> {
//      let mm  = tensor::matmul(x, w);
//      let out = tensor::relu(mm);
//      return out;
//  }
 
TEST(IntegrationTest, TensorMatMulRelu)
{
    IRModule mod("nn.tcc");
 
    auto t_f32   = Type::tensor(Type::f32());
    auto fn_type = Type::fn({t_f32, t_f32}, t_f32);
    auto* fn     = mod.add_function("@forward", fn_type);
    auto* x_arg  = fn->add_param("%x", t_f32);
    auto* w_arg  = fn->add_param("%w", t_f32);
    auto* entry  = fn->create_entry();
 
    auto x_val = std::shared_ptr<Value>(x_arg, [](Value*){});
    auto w_val = std::shared_ptr<Value>(w_arg, [](Value*){});
 
    auto* mm = entry->emit<TensorOpInst>(
        "%mm", t_f32, TensorOpCode::MatMul,
        std::vector<ValuePtr>{x_val, w_val});
 
    auto mm_val = std::shared_ptr<Value>(mm, [](Value*){});
    auto* relu  = entry->emit<TensorOpInst>(
        "%out", t_f32, TensorOpCode::Relu,
        std::vector<ValuePtr>{mm_val});
 
    auto relu_val = std::shared_ptr<Value>(relu, [](Value*){});
    entry->emit<ReturnInst>(relu_val);
 
    EXPECT_EQ(entry->insts.size(), 3u);
    EXPECT_TRUE(x_val->has_uses());
    EXPECT_TRUE(w_val->has_uses());
    EXPECT_TRUE(mm_val->has_uses());  // relu consumes mm
 
    // Verify opcode chain
    auto* mm_op   = dynamic_cast<TensorOpInst*>(entry->insts[0].get());
    auto* relu_op = dynamic_cast<TensorOpInst*>(entry->insts[1].get());
    ASSERT_NE(mm_op,   nullptr);
    ASSERT_NE(relu_op, nullptr);
    EXPECT_EQ(mm_op->op,   TensorOpCode::MatMul);
    EXPECT_EQ(relu_op->op, TensorOpCode::Relu);
}
 
// ─── 3. If-else control flow ─────────────────────────────────────────────────
//
//  fn abs_val(x: f32) -> f32 {
//      if (x > 0.0) { return x; } else { return -x; }
//  }
 
TEST(IntegrationTest, IfElseCFG)
{
    IRModule mod("cfgtest.tcc");
 
    auto fn_type = Type::fn({Type::f32()}, Type::f32());
    auto* fn     = mod.add_function("@abs_val", fn_type);
    auto* x_arg  = fn->add_param("%x", Type::f32());
    auto* entry  = fn->create_entry();
    auto* if_true  = fn->add_block("if.true");
    auto* if_false = fn->add_block("if.false");
 
    auto x_val  = std::shared_ptr<Value>(x_arg, [](Value*){});
    auto zero   = std::make_shared<ConstantFloat>(0.0, Type::f32());
 
    // entry: %cond = x > 0.0; br %cond, if.true, if.false
    auto* cmp = entry->emit<CmpInst>("%cond", CmpCode::Gt, x_val, zero);
    auto cmp_val = std::shared_ptr<Value>(cmp, [](Value*){});
    entry->emit<CondBranchInst>(cmp_val, if_true, if_false);
 
    // if.true: ret x
    if_true->emit<ReturnInst>(x_val);
 
    // if.false: %neg = -x; ret %neg
    auto* neg = if_false->emit<UnOpInst>("%neg", Type::f32(), UnOpCode::FNeg, x_val);
    auto neg_val = std::shared_ptr<Value>(neg, [](Value*){});
    if_false->emit<ReturnInst>(neg_val);
 
    EXPECT_EQ(fn->blocks.size(), 3u);
    EXPECT_TRUE(entry->is_terminated());
    EXPECT_TRUE(if_true->is_terminated());
    EXPECT_TRUE(if_false->is_terminated());
 
    // cond and x each used in two places
    EXPECT_EQ(x_val->uses.size(),   3u);  // cmp + true_ret + neg
    EXPECT_EQ(cmp_val->uses.size(), 1u);  // cond_br
}
 
// ─── 4. Async spawn + await ──────────────────────────────────────────────────
//
//  async fn parallel_add(a: Tensor<f32>, b: Tensor<f32>) -> Tensor<f32> {
//      let h_a = spawn tensor::sum(a);
//      let h_b = spawn tensor::sum(b);
//      return await h_a + await h_b;
//  }
 
TEST(IntegrationTest, AsyncSpawnAwait)
{
    IRModule mod("async.tcc");
 
    auto t_f32   = Type::tensor(Type::f32());
    auto fn_type = Type::fn({t_f32, t_f32}, Type::f32());
    auto* fn     = mod.add_function("@parallel_add", fn_type, /*is_async=*/true);
    auto* a_arg  = fn->add_param("%a", t_f32);
    auto* b_arg  = fn->add_param("%b", t_f32);
    auto* entry  = fn->create_entry();
 
    auto a_val = std::shared_ptr<Value>(a_arg, [](Value*){});
    auto b_val = std::shared_ptr<Value>(b_arg, [](Value*){});
 
    // sum tasks (represented as TensorOpInst values passed into SpawnInst)
    auto* sum_a_op = entry->emit<TensorOpInst>(
        "%sum_a_task", Type::f32(), TensorOpCode::Sum,
        std::vector<ValuePtr>{a_val});
    auto* sum_b_op = entry->emit<TensorOpInst>(
        "%sum_b_task", Type::f32(), TensorOpCode::Sum,
        std::vector<ValuePtr>{b_val});
 
    auto sum_a_v = std::shared_ptr<Value>(sum_a_op, [](Value*){});
    auto sum_b_v = std::shared_ptr<Value>(sum_b_op, [](Value*){});
 
    auto* h_a = entry->emit<SpawnInst>("%h_a", Type::infer(), sum_a_v);
    auto* h_b = entry->emit<SpawnInst>("%h_b", Type::infer(), sum_b_v);
 
    auto h_a_v = std::shared_ptr<Value>(h_a, [](Value*){});
    auto h_b_v = std::shared_ptr<Value>(h_b, [](Value*){});
 
    auto* r_a = entry->emit<AwaitInst>("%r_a", Type::f32(), h_a_v);
    auto* r_b = entry->emit<AwaitInst>("%r_b", Type::f32(), h_b_v);
 
    auto r_a_v = std::shared_ptr<Value>(r_a, [](Value*){});
    auto r_b_v = std::shared_ptr<Value>(r_b, [](Value*){});
 
    auto* total = entry->emit<BinOpInst>("%total", Type::f32(), BinOpCode::FAdd, r_a_v, r_b_v);
    auto total_v = std::shared_ptr<Value>(total, [](Value*){});
    entry->emit<ReturnInst>(total_v);
 
    EXPECT_TRUE(fn->is_async);
    EXPECT_EQ(entry->insts.size(), 8u);  // sum_a, sum_b, spawn×2, await×2, fadd, ret
    EXPECT_TRUE(entry->is_terminated());
    EXPECT_TRUE(h_a_v->has_uses());  // await consumes handle
    EXPECT_TRUE(h_b_v->has_uses());
}
 
// ─── 5. Residual bottleneck (stress) ─────────────────────────────────────────
//
//  async fn residual_bottleneck(x, w1, w2) -> Tensor<f32> {
//      let h_a  = spawn matmul(x, w1)
//      let h_b  = spawn matmul(x, w2)
//      let out  = relu(await h_a + await h_b)
//      return out + x
//  }
 
TEST(IntegrationTest, ResidualBottleneck)
{
    IRModule mod("residual.tcc");
 
    auto t_f32   = Type::tensor(Type::f32());
    auto fn_type = Type::fn({t_f32, t_f32, t_f32}, t_f32);
    auto* fn     = mod.add_function("@residual_bottleneck", fn_type, /*is_async=*/true);
 
    auto* x_arg  = fn->add_param("%x",  t_f32);
    auto* w1_arg = fn->add_param("%w1", t_f32);
    auto* w2_arg = fn->add_param("%w2", t_f32);
    auto* entry  = fn->create_entry();
 
    auto x_v  = std::shared_ptr<Value>(x_arg,  [](Value*){});
    auto w1_v = std::shared_ptr<Value>(w1_arg, [](Value*){});
    auto w2_v = std::shared_ptr<Value>(w2_arg, [](Value*){});
 
    // branch_a = matmul(x, w1)
    auto* mm1 = entry->emit<TensorOpInst>(
        "%mm1", t_f32, TensorOpCode::MatMul,
        std::vector<ValuePtr>{x_v, w1_v});
    auto mm1_v = std::shared_ptr<Value>(mm1, [](Value*){});
 
    // branch_b = matmul(x, w2)
    auto* mm2 = entry->emit<TensorOpInst>(
        "%mm2", t_f32, TensorOpCode::MatMul,
        std::vector<ValuePtr>{x_v, w2_v});
    auto mm2_v = std::shared_ptr<Value>(mm2, [](Value*){});
 
    // spawn both branches
    auto* h_a = entry->emit<SpawnInst>("%h_a", Type::infer(), mm1_v);
    auto* h_b = entry->emit<SpawnInst>("%h_b", Type::infer(), mm2_v);
    auto h_a_v = std::shared_ptr<Value>(h_a, [](Value*){});
    auto h_b_v = std::shared_ptr<Value>(h_b, [](Value*){});
 
    // await + add
    auto* r_a = entry->emit<AwaitInst>("%r_a", t_f32, h_a_v);
    auto* r_b = entry->emit<AwaitInst>("%r_b", t_f32, h_b_v);
    auto r_a_v = std::shared_ptr<Value>(r_a, [](Value*){});
    auto r_b_v = std::shared_ptr<Value>(r_b, [](Value*){});
 
    auto* branch_sum = entry->emit<BinOpInst>(
        "%branch_sum", t_f32, BinOpCode::FAdd, r_a_v, r_b_v);
    auto bsum_v = std::shared_ptr<Value>(branch_sum, [](Value*){});
 
    // relu
    auto* relu = entry->emit<TensorOpInst>(
        "%out", t_f32, TensorOpCode::Relu,
        std::vector<ValuePtr>{bsum_v});
    auto relu_v = std::shared_ptr<Value>(relu, [](Value*){});
 
    // residual: out + x
    auto* residual = entry->emit<BinOpInst>(
        "%residual", t_f32, BinOpCode::FAdd, relu_v, x_v);
    auto res_v = std::shared_ptr<Value>(residual, [](Value*){});
 
    entry->emit<ReturnInst>(res_v);
 
    // ── Structural assertions ─────────────────────────────────────────────────
    EXPECT_TRUE(fn->is_async);
    EXPECT_EQ(fn->params.size(), 3u);
    EXPECT_EQ(entry->insts.size(), 10u);
    EXPECT_TRUE(entry->is_terminated());
 
    // x is used by both matmuls AND the final residual add
    EXPECT_EQ(x_v->uses.size(), 3u);
 
    // both handles are consumed exactly once (by their awaits)
    EXPECT_EQ(h_a_v->uses.size(), 1u);
    EXPECT_EQ(h_b_v->uses.size(), 1u);
 
    // relu output consumed by residual add
    EXPECT_TRUE(relu_v->has_uses());
}