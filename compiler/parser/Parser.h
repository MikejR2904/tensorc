#pragma once

#include <string>
#include <memory>
#include <stdexcept>
#include <unordered_set>
#include "../lexer/Lexer.h"
#include "../ast/ASTNode.h"

// Throw parsing error if there's syntax mistake
struct ParseError : public std::runtime_error
{
    Position pos;
    ParseError(const std::string& msg, Position p)
        : std::runtime_error(msg), pos(p) {}
};

class Parser
{
public:
    explicit Parser(Lexer& lexer);
    Program parse();

private:
    Lexer& lexer;
    Token  current;
    Token  previous;

    std::unordered_set<std::string> active_generic_names;
    std::optional<GenericParams> last_tensor_gp;

    void      advance();
    bool      check(TokenKind k) const;
    bool      match(TokenKind k);
    void      expect(TokenKind k, const std::string& msg);
    void      consumeOptionalSemicolon();
    ParseError error(const std::string& msg) const;

    StmtPtr parseStatement();
    StmtPtr parseLet();
    StmtPtr parseFnDecl();
    StmtPtr parseReturn();
    StmtPtr parseIf();
    StmtPtr parseFor();
    StmtPtr parseWhile();
    StmtPtr parseMatch();
    StmtPtr parseImport();
    StmtPtr parseSpawn();
    StmtPtr parseBreak();
    StmtPtr parseContinue();
    StmtPtr parseStruct();

    Compound parseCompound();
    ExprPtr parseExpr();
    ExprPtr parsePipe();            // |>      lowest
    ExprPtr parseAssign();          // =  +=  -=  *=  /=
    ExprPtr parseOr();              // ||
    ExprPtr parseAnd();             // &&
    ExprPtr parseEquality();        // ==  !=
    ExprPtr parseComparison();      // <  >  <=  >=
    ExprPtr parseAdditive();        // +  -
    ExprPtr parseMultiplicative();  // *  /
    ExprPtr parseMatMul();          // @
    ExprPtr parseUnary();           // !  -
    ExprPtr parsePostfix();         // call  index  field  scope  <-
    ExprPtr parsePrimary();         // literals  identifiers  keywords
    void parseShapeAnnotation();
    ExprPtr parseIntLit();
    ExprPtr parseFloatLit();
    ExprPtr parseBoolLit();
    ExprPtr parseStringLit();
    ExprPtr parseIdentOrCall();
    ExprPtr parseGrouped();
    ExprPtr parseIfExpr();
    ExprPtr parseMatchExpr();
    ExprPtr parseFnExpr();
    ExprPtr parseSpawnExpr();
    ExprPtr parseBlockExpr();
    ExprPtr parseAwait();
    ExprPtr parseGrad();
    ExprPtr parseChannelSend(ExprPtr lhs);
    ExprPtr parseArrayLit();
    ExprPtr parseTensorLit();
    ExprPtr parseSetLit();
    ExprPtr parseMapLit();
    ExprPtr parseQueueLit();
    ExprPtr parseStackLit();
    ExprPtr parseTupleLit();
    ExprPtr parseStructLit(const std::string& name, Position p);
    std::vector<std::vector<ExprPtr>> parseTensorRows();
    TyKind       parseType();
    GenericParams parseGenericParams();
    std::vector<std::string> parseGenericNames();
    std::vector<ExprPtr>  parseExprList(TokenKind terminator);
    std::vector<Ident>    parseParamList();
    MatchArm              parseMatchArm();
    StmtPtr makeStmt(StmtKind kind, Position pos);
    ExprPtr makeExpr(ExprKind kind, Position pos);
};