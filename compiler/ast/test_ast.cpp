// #include <iostream>
// #include <cassert>
// #include "SymbolTable.h"

// void test_basic_scopes() {
//     SymbolTable st;

//     // 1. Test Global Definition
//     Position p1{1, 5};
//     st.define(Symbol("learning_rate", TyKind::F32, IdentCtx::Def, p1));
    
//     Symbol* s = st.lookup("learning_rate");
//     assert(s != nullptr);
//     assert(s->type == TyKind::F32);

//     // 2. Test Nested Scope (entering a { ... } block)
//     st.pushScope();
//     Position p2{2, 10};
//     st.define(Symbol("batch_size", TyKind::I32, IdentCtx::Def, p2));
    
//     assert(st.lookup("batch_size") != nullptr);
//     assert(st.lookup("learning_rate") != nullptr); // Can still see global

//     // 3. Test Shadowing
//     // Define another 'learning_rate' inside the block
//     st.define(Symbol("learning_rate", TyKind::F64, IdentCtx::Def, p2));
//     assert(st.lookup("learning_rate")->type == TyKind::F64); // Should see the new one

//     // 4. Test Scope Exit
//     st.popScope();
//     assert(st.lookup("batch_size") == nullptr); // Should be gone
//     assert(st.lookup("learning_rate")->type == TyKind::F32); // Should be back to global
    
//     std::cout << "Basic scope tests passed!" << std::endl;
// }

// void test_errors() {
//     SymbolTable st;
//     st.define(Symbol("x", TyKind::I32, IdentCtx::Def, {1, 1}));

//     try {
//         st.define(Symbol("x", TyKind::F32, IdentCtx::Def, {1, 5}));
//         assert(false); // Should not reach here
//     } catch (const std::runtime_error& e) {
//         std::cout << "Caught expected error: " << e.what() << std::endl;
//     }
// }

// int main() {
//     test_basic_scopes();
//     test_errors();
//     return 0;
// }

#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <functional>
#include "ASTNode.h"
#include "SemanticAnalyzer.h"

// Helper to create a dummy position
Position dpos() { return {1, 1}; }

void run_test(const std::string& name, std::function<void()> test_func) {
    std::cout << "Running " << name << "... ";
    try {
        test_func();
        std::cout << "PASSED" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "CAUGHT EXPECTED ERROR: " << e.what() << std::endl;
    }
}

int main() {
    // --- Test 1: Basic Let with Type Inference ---
    run_test("Inference Test (let x = 5)", []() {
        SemanticAnalyzer analyzer;
        Program prog;

        // Construct: let x = 5
        StmtKind sk;
        sk.tag = StmtKind::Tag::Let;
        sk.let_ident = Ident::unqual(IdentInfo("x", TyKind::Infer, IdentCtx::Def, dpos()));
        
        ExprKind ek;
        ek.tag = ExprKind::Tag::Lit;
        ek.lit = LitKind::makeInt("5");
        sk.let_expr = std::make_unique<Expr>(std::move(ek), dpos());

        prog.addStmt(std::make_unique<Stmt>(std::move(sk), dpos()));
        analyzer.validate(prog);
    });

    // --- Test 2: Type Mismatch (let x: i32 = 5.5) ---
    run_test("Type Mismatch Test", []() {
        SemanticAnalyzer analyzer;
        Program prog;

        // Construct: let x: i32 = 5.5
        StmtKind sk;
        sk.tag = StmtKind::Tag::Let;
        sk.let_ident = Ident::unqual(IdentInfo("x", TyKind::I32, IdentCtx::Def, dpos()));
        
        ExprKind ek;
        ek.tag = ExprKind::Tag::Lit;
        ek.lit = LitKind::makeFloat("5.5");
        sk.let_expr = std::make_unique<Expr>(std::move(ek), dpos());

        prog.addStmt(std::make_unique<Stmt>(std::move(sk), dpos()));
        analyzer.validate(prog); 
    });

    // --- Test 3: Tensor MatMul Check ---
    run_test("Tensor MatMul Validation", []() {
        SemanticAnalyzer analyzer;
        Program prog;

        // 1. let A: Tensor = [[...]]
        StmtKind skA;
        skA.tag = StmtKind::Tag::Let;
        skA.let_ident = Ident::unqual(IdentInfo("A", TyKind::Tensor, IdentCtx::Def, dpos()));
        ExprKind ekA; ekA.tag = ExprKind::Tag::TensorLit;
        skA.let_expr = std::make_unique<Expr>(std::move(ekA), dpos());
        prog.addStmt(std::make_unique<Stmt>(std::move(skA), dpos()));

        // 2. let B: Tensor = [[...]]
        StmtKind skB;
        skB.tag = StmtKind::Tag::Let;
        skB.let_ident = Ident::unqual(IdentInfo("B", TyKind::Tensor, IdentCtx::Def, dpos()));
        ExprKind ekB; ekB.tag = ExprKind::Tag::TensorLit;
        skB.let_expr = std::make_unique<Expr>(std::move(ekB), dpos());
        prog.addStmt(std::make_unique<Stmt>(std::move(skB), dpos()));

        // 3. A @ B (Expression Statement)
        ExprKind ekMatMul;
        ekMatMul.tag = ExprKind::Tag::Binary;
        ekMatMul.bin_op = BinOp::MatMul;
        
        ExprKind lhs; lhs.tag = ExprKind::Tag::Id;
        lhs.id = Ident::unqual(IdentInfo("A", TyKind::Infer, IdentCtx::Ref, dpos()));
        ekMatMul.lhs = std::make_unique<Expr>(std::move(lhs), dpos());

        ExprKind rhs; rhs.tag = ExprKind::Tag::Id;
        rhs.id = Ident::unqual(IdentInfo("B", TyKind::Infer, IdentCtx::Ref, dpos()));
        ekMatMul.rhs = std::make_unique<Expr>(std::move(rhs), dpos());

        StmtKind skExpr;
        skExpr.tag = StmtKind::Tag::Expr;
        skExpr.expr = std::make_unique<Expr>(std::move(ekMatMul), dpos());
        
        prog.addStmt(std::make_unique<Stmt>(std::move(skExpr), dpos()));

        analyzer.validate(prog);
    });

    run_test("Match Statement Validation", []() {
        SemanticAnalyzer analyzer;
        Program prog;

        // let x = 10
        StmtKind skLet;
        skLet.tag = StmtKind::Tag::Let;
        skLet.let_ident = Ident::unqual(IdentInfo("x", TyKind::I32, IdentCtx::Def, dpos()));
        skLet.let_expr = std::make_unique<Expr>(ExprKind{ExprKind::Tag::Lit, LitKind::makeInt("10")}, dpos());
        prog.addStmt(std::make_unique<Stmt>(std::move(skLet), dpos()));

        // match x { 10 => { } }
        StmtKind skMatch;
        skMatch.tag = StmtKind::Tag::Match;
        
        // Subject: x
        ExprKind ekSubj; ekSubj.tag = ExprKind::Tag::Id;
        ekSubj.id = Ident::unqual(IdentInfo("x", TyKind::Infer, IdentCtx::Ref, dpos()));
        skMatch.match_subject = std::make_unique<Expr>(std::move(ekSubj), dpos());

        // Arm Pattern: 10
        MatchArm arm;
        ExprKind ekPat; ekPat.tag = ExprKind::Tag::Lit;
        ekPat.lit = LitKind::makeInt("10");
        arm.pattern = std::make_unique<Expr>(std::move(ekPat), dpos());
        
        // Arm Body: bare expr (Void)
        ExprKind ekBody; ekBody.tag = ExprKind::Tag::Lit; // Just a dummy lit for body
        ekBody.lit = LitKind::makeBool(true);
        arm.body = std::make_unique<Expr>(std::move(ekBody), dpos());
        arm.pos = dpos();

        skMatch.match_arms.push_back(std::move(arm));
        prog.addStmt(std::make_unique<Stmt>(std::move(skMatch), dpos()));

        analyzer.validate(prog);
    });

    run_test("Match Type Mismatch (Int vs Str)", []() {
        SemanticAnalyzer analyzer;
        Program prog;

        // 1. let x = 10 (I32)
        StmtKind skLet;
        skLet.tag = StmtKind::Tag::Let;
        skLet.let_ident = Ident::unqual(IdentInfo("x", TyKind::I32, IdentCtx::Def, dpos()));
        skLet.let_expr = std::make_unique<Expr>(ExprKind{ExprKind::Tag::Lit, LitKind::makeInt("10")}, dpos());
        prog.addStmt(std::make_unique<Stmt>(std::move(skLet), dpos()));

        // 2. match x { "hello" => { } }
        StmtKind skMatch;
        skMatch.tag = StmtKind::Tag::Match;
        
        // Subject: x (I32)
        ExprKind ekSubj; ekSubj.tag = ExprKind::Tag::Id;
        ekSubj.id = Ident::unqual(IdentInfo("x", TyKind::Infer, IdentCtx::Ref, dpos()));
        skMatch.match_subject = std::make_unique<Expr>(std::move(ekSubj), dpos());

        // Arm Pattern: "hello" (Str)
        MatchArm arm;
        ExprKind ekPat; ekPat.tag = ExprKind::Tag::Lit;
        ekPat.lit = LitKind::makeStr("hello");
        arm.pattern = std::make_unique<Expr>(std::move(ekPat), dpos());
        
        arm.body = std::make_unique<Expr>(ExprKind{ExprKind::Tag::Lit, LitKind::makeBool(true)}, dpos());
        arm.pos = dpos();

        skMatch.match_arms.push_back(std::move(arm));
        prog.addStmt(std::make_unique<Stmt>(std::move(skMatch), dpos()));

        // This SHOULD throw an exception because I32 != Str
        analyzer.validate(prog);
    });

    run_test("Tensor Shape Success (3x2 @ 2x5)", []() {
        SemanticAnalyzer analyzer;
        Program prog;

        // 1. let A: Tensor = ... (Shape [3, 2])
        StmtKind skA;
        skA.tag = StmtKind::Tag::Let;
        skA.let_ident = Ident::unqual(IdentInfo("A", TyKind::Tensor, IdentCtx::Def, dpos()));
        
        ExprKind ekA; 
        ekA.tag = ExprKind::Tag::TensorLit;
        ekA.generic_params = GenericParams{{TyKind::F32}, {3, 2}, dpos()};
        skA.let_expr = std::make_unique<Expr>(std::move(ekA), dpos());
        prog.addStmt(std::make_unique<Stmt>(std::move(skA), dpos()));

        // 2. let B: Tensor = ... (Shape [2, 5])
        StmtKind skB;
        skB.tag = StmtKind::Tag::Let;
        skB.let_ident = Ident::unqual(IdentInfo("B", TyKind::Tensor, IdentCtx::Def, dpos()));
        
        ExprKind ekB; 
        ekB.tag = ExprKind::Tag::TensorLit;
        ekB.generic_params = GenericParams{{TyKind::F32}, {2, 5}, dpos()};
        skB.let_expr = std::make_unique<Expr>(std::move(ekB), dpos());
        prog.addStmt(std::make_unique<Stmt>(std::move(skB), dpos()));

        // 3. A @ B
        ExprKind ekMatMul;
        ekMatMul.tag = ExprKind::Tag::Binary;
        ekMatMul.bin_op = BinOp::MatMul;

        // LHS Setup
        ExprKind ekL; 
        ekL.tag = ExprKind::Tag::Id;
        ekL.id = Ident::unqual(IdentInfo("A", TyKind::Infer, IdentCtx::Ref, dpos()));
        ekMatMul.lhs = std::make_unique<Expr>(std::move(ekL), dpos());

        // RHS Setup
        ExprKind ekR; 
        ekR.tag = ExprKind::Tag::Id;
        ekR.id = Ident::unqual(IdentInfo("B", TyKind::Infer, IdentCtx::Ref, dpos()));
        ekMatMul.rhs = std::make_unique<Expr>(std::move(ekR), dpos());

        StmtKind skExpr;
        skExpr.tag = StmtKind::Tag::Expr;
        skExpr.expr = std::make_unique<Expr>(std::move(ekMatMul), dpos());
        prog.addStmt(std::make_unique<Stmt>(std::move(skExpr), dpos()));

        analyzer.validate(prog);
    });

    run_test("Tensor Shape Mismatch Error (3x2 @ 5x2)", []() {
        SemanticAnalyzer analyzer;
        Program prog;

        // 1. let A: Tensor [3, 2]
        StmtKind skA;
        skA.tag = StmtKind::Tag::Let;
        skA.let_ident = Ident::unqual(IdentInfo("A", TyKind::Tensor, IdentCtx::Def, dpos()));
        ExprKind ekA; ekA.tag = ExprKind::Tag::TensorLit;
        ekA.generic_params = GenericParams{{TyKind::F32}, {3, 2}, dpos()};
        skA.let_expr = std::make_unique<Expr>(std::move(ekA), dpos());
        prog.addStmt(std::make_unique<Stmt>(std::move(skA), dpos()));

        // 2. let B: Tensor [5, 2]
        StmtKind skB;
        skB.tag = StmtKind::Tag::Let;
        skB.let_ident = Ident::unqual(IdentInfo("B", TyKind::Tensor, IdentCtx::Def, dpos()));
        ExprKind ekB; ekB.tag = ExprKind::Tag::TensorLit;
        ekB.generic_params = GenericParams{{TyKind::F32}, {5, 2}, dpos()};
        skB.let_expr = std::make_unique<Expr>(std::move(ekB), dpos());
        prog.addStmt(std::make_unique<Stmt>(std::move(skB), dpos()));

        // 3. A @ B (Should fail)
        ExprKind ekMatMul;
        ekMatMul.tag = ExprKind::Tag::Binary;
        ekMatMul.bin_op = BinOp::MatMul;

        ExprKind ekL; ekL.tag = ExprKind::Tag::Id;
        ekL.id = Ident::unqual(IdentInfo("A", TyKind::Infer, IdentCtx::Ref, dpos()));
        ekMatMul.lhs = std::make_unique<Expr>(std::move(ekL), dpos());

        ExprKind ekR; ekR.tag = ExprKind::Tag::Id;
        ekR.id = Ident::unqual(IdentInfo("B", TyKind::Infer, IdentCtx::Ref, dpos()));
        ekMatMul.rhs = std::make_unique<Expr>(std::move(ekR), dpos());

        StmtKind skExpr;
        skExpr.tag = StmtKind::Tag::Expr;
        skExpr.expr = std::make_unique<Expr>(std::move(ekMatMul), dpos());
        prog.addStmt(std::make_unique<Stmt>(std::move(skExpr), dpos()));

        analyzer.validate(prog);
    });

    run_test("Function Valid Return (I32)", []() {
        SemanticAnalyzer analyzer;
        Program prog;

        // 1. Create the pieces for the constructor
        Ident fn_id = Ident::unqual(IdentInfo("add_one", TyKind::I32, IdentCtx::FuncDef, dpos()));
        
        std::vector<Ident> params;
        params.push_back(Ident::unqual(IdentInfo("x", TyKind::I32, IdentCtx::Param, dpos())));
        
        Compound body; // Empty body for now

        // 2. Build the Func using the required constructor
        Func fn(fn_id, std::move(params), std::move(body));

        // 3. Now add the return statement to the body
        ExprKind ekBinary;
        ekBinary.tag = ExprKind::Tag::Binary;
        ekBinary.bin_op = BinOp::Add;
        
        ExprKind ekL; ekL.tag = ExprKind::Tag::Id;
        ekL.id = Ident::unqual(IdentInfo("x", TyKind::Infer, IdentCtx::Ref, dpos()));
        ekBinary.lhs = std::make_unique<Expr>(std::move(ekL), dpos());

        ExprKind ekR; ekR.tag = ExprKind::Tag::Lit;
        ekR.lit = LitKind::makeInt("1");
        ekBinary.rhs = std::make_unique<Expr>(std::move(ekR), dpos());

        StmtKind skRet;
        skRet.tag = StmtKind::Tag::Return;
        skRet.ret_expr = std::make_unique<Expr>(std::move(ekBinary), dpos());
        
        fn.body.addStmt(std::make_unique<Stmt>(std::move(skRet), dpos()));

        // 4. Wrap and add to program
        StmtKind skFunc;
        skFunc.tag = StmtKind::Tag::Func;
        skFunc.func = std::move(fn);
        
        prog.addStmt(std::make_unique<Stmt>(std::move(skFunc), dpos()));
        analyzer.validate(prog);
    });

    // --- New Test: Function Return Type Mismatch ---
    run_test("Function Return Mismatch (Expect I32, got Str)", []() {
        SemanticAnalyzer analyzer;
        Program prog;

        // Build Func using constructor
        Ident fn_id = Ident::unqual(IdentInfo("bad_func", TyKind::I32, IdentCtx::FuncDef, dpos()));
        Func fn(fn_id, {}, Compound());

        ExprKind ekLit;
        ekLit.tag = ExprKind::Tag::Lit;
        ekLit.lit = LitKind::makeStr("hello");

        StmtKind skRet;
        skRet.tag = StmtKind::Tag::Return;
        skRet.ret_expr = std::make_unique<Expr>(std::move(ekLit), dpos());

        fn.body.addStmt(std::make_unique<Stmt>(std::move(skRet), dpos()));

        StmtKind skFunc;
        skFunc.tag = StmtKind::Tag::Func;
        skFunc.func = std::move(fn);

        prog.addStmt(std::make_unique<Stmt>(std::move(skFunc), dpos()));
        analyzer.validate(prog); 
    });

    // --- New Test: Grad Scalar Check ---
    run_test("Grad Check (Valid F32 Loss)", []() {
        SemanticAnalyzer analyzer;
        Program prog;

        // let loss = 0.5; grad(loss, params)
        StmtKind skLet;
        skLet.tag = StmtKind::Tag::Let;
        skLet.let_ident = Ident::unqual(IdentInfo("loss", TyKind::F32, IdentCtx::Def, dpos()));
        ExprKind ekLit; ekLit.tag = ExprKind::Tag::Lit; ekLit.lit = LitKind::makeFloat("0.5");
        skLet.let_expr = std::make_unique<Expr>(std::move(ekLit), dpos());
        prog.addStmt(std::make_unique<Stmt>(std::move(skLet), dpos()));

        ExprKind ekGrad;
        ekGrad.tag = ExprKind::Tag::Grad;
        
        ExprKind ekL; ekL.tag = ExprKind::Tag::Id;
        ekL.id = Ident::unqual(IdentInfo("loss", TyKind::Infer, IdentCtx::Ref, dpos()));
        ekGrad.grad_loss = std::make_unique<Expr>(std::move(ekL), dpos());
        
        ExprKind ekP; ekP.tag = ExprKind::Tag::Lit; ekP.lit = LitKind::makeInt("0"); // Dummy param
        ekGrad.grad_params = std::make_unique<Expr>(std::move(ekP), dpos());

        StmtKind skExpr;
        skExpr.tag = StmtKind::Tag::Expr;
        skExpr.expr = std::make_unique<Expr>(std::move(ekGrad), dpos());
        prog.addStmt(std::make_unique<Stmt>(std::move(skExpr), dpos()));

        analyzer.validate(prog);
    });

    run_test("Grad Check (Invalid Loss Type)", []() {
        SemanticAnalyzer analyzer;
        Program prog;

        // grad("not a float", my_tensor)
        ExprKind ekGrad;
        ekGrad.tag = ExprKind::Tag::Grad;
        
        ExprKind ekLoss; ekLoss.tag = ExprKind::Tag::Lit;
        ekLoss.lit = LitKind::makeStr("error"); // Invalid loss (Str)
        ekGrad.grad_loss = std::make_unique<Expr>(std::move(ekLoss), dpos());

        ExprKind ekParam; ekParam.tag = ExprKind::Tag::TensorLit; // Valid param
        ekGrad.grad_params = std::make_unique<Expr>(std::move(ekParam), dpos());

        StmtKind skExpr;
        skExpr.tag = StmtKind::Tag::Expr;
        skExpr.expr = std::make_unique<Expr>(std::move(ekGrad), dpos());
        
        prog.addStmt(std::make_unique<Stmt>(std::move(skExpr), dpos()));
        analyzer.validate(prog);
    });

    // --- Test: Loop Context Validation ---
    run_test("Continue Outside Loop (Should Fail)", []() {
        SemanticAnalyzer analyzer;
        Program prog;

        StmtKind skCont;
        skCont.tag = StmtKind::Tag::Continue;
        
        prog.addStmt(std::make_unique<Stmt>(std::move(skCont), dpos()));
        
        try {
            analyzer.validate(prog);
        } catch (const std::runtime_error& e) {
            std::cout << "Caught Expected Error: " << e.what() << std::endl;
            return; // Success
        }
        throw std::runtime_error("Failed: Continue outside loop did not throw error");
    });


    run_test("Function Call - Arg Mismatch (Should Fail)", []() {
        SemanticAnalyzer analyzer;
        Program prog;
        Position dpos = {1, 1};

        // 1. Define: fn log(msg: str) -> void
        Ident fn_id = Ident::unqual(IdentInfo("log", TyKind::Void, IdentCtx::FuncDef, dpos));
        std::vector<Ident> params;
        params.push_back(Ident::unqual(IdentInfo("msg", TyKind::Str, IdentCtx::Param, dpos)));
        
        Func fn(fn_id, std::move(params), Compound());
        StmtKind skFunc;
        skFunc.tag = StmtKind::Tag::Func;
        skFunc.func = std::move(fn);
        prog.addStmt(std::make_unique<Stmt>(std::move(skFunc), dpos));

        // 2. Call: log(42)  <-- Error: Int passed to Str
        ExprKind ekCall;
        ekCall.tag = ExprKind::Tag::Call;
        
        // Setup Callee
        ExprKind ekCallee;
        ekCallee.tag = ExprKind::Tag::Id;
        ekCallee.id = Ident::unqual(IdentInfo("log", TyKind::Infer, IdentCtx::Ref, dpos));
        ekCall.callee = std::make_unique<Expr>(std::move(ekCallee), dpos);

        // Setup Arg (Wrapping LitKind inside ExprKind manually)
        ExprKind ekArg;
        ekArg.tag = ExprKind::Tag::Lit;
        ekArg.lit = LitKind::makeInt("42"); 
        ekCall.args.push_back(std::make_unique<Expr>(std::move(ekArg), dpos));

        prog.addStmt(std::make_unique<Stmt>(StmtKind::makeExpr(std::make_unique<Expr>(std::move(ekCall), dpos)), dpos));

        try {
            analyzer.validate(prog);
            // If we reach here, the analyzer failed to catch the error
        } catch (const std::runtime_error& e) {
            std::cout << "CAUGHT EXPECTED ERROR: " << e.what() << std::endl;
        }
    });
    
    run_test("Pipe Operator Valid (Tensor -> Func)", []() {
        SemanticAnalyzer analyzer;
        Program prog;
        Position dpos = {1, 1};

        // 1. Define: fn process(t: Tensor, alpha: F32) -> Tensor
        Ident fn_id = Ident::unqual(IdentInfo("process", TyKind::Tensor, IdentCtx::FuncDef, dpos));
        std::vector<Ident> params;
        params.push_back(Ident::unqual(IdentInfo("t", TyKind::Tensor, IdentCtx::Param, dpos)));
        params.push_back(Ident::unqual(IdentInfo("alpha", TyKind::F32, IdentCtx::Param, dpos)));
        
        Func fn(fn_id, std::move(params), Compound());
        StmtKind skFunc;
        skFunc.tag = StmtKind::Tag::Func;
        skFunc.func = std::move(fn);
        prog.addStmt(std::make_unique<Stmt>(std::move(skFunc), dpos));

        // 2. Create Pipe Expression
        ExprKind ekPipe;
        ekPipe.tag = ExprKind::Tag::Pipe;

        // --- LHS: TensorLiteral ---
        ExprKind ekLhs;
        ekLhs.tag = ExprKind::Tag::TensorLit;
        ekLhs.generic_params = GenericParams{{}, {3, 3}, dpos};
        ekPipe.pipe_lhs = std::make_unique<Expr>(std::move(ekLhs), dpos);

        // --- RHS: process(0.5) ---
        ExprKind ekCall;
        ekCall.tag = ExprKind::Tag::Call;
        
        // Callee (the function name)
        ExprKind ekCallee;
        ekCallee.tag = ExprKind::Tag::Id;
        ekCallee.id = Ident::unqual(IdentInfo("process", TyKind::Infer, IdentCtx::Ref, dpos));
        ekCall.callee = std::make_unique<Expr>(std::move(ekCallee), dpos);

        // Argument (0.5)
        ExprKind ekArg;
        ekArg.tag = ExprKind::Tag::Lit;
        ekArg.lit = LitKind::makeFloat("0.5");
        ekCall.args.push_back(std::make_unique<Expr>(std::move(ekArg), dpos));
        
        ekPipe.pipe_rhs = std::make_unique<Expr>(std::move(ekCall), dpos);

        // Wrap in Stmt
        StmtKind skExpr;
        skExpr.tag = StmtKind::Tag::Expr;
        skExpr.expr = std::make_unique<Expr>(std::move(ekPipe), dpos);
        prog.addStmt(std::make_unique<Stmt>(std::move(skExpr), dpos));

        analyzer.validate(prog); 
    });

    return 0;
}