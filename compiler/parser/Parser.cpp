#include "Parser.h"
#include <sstream>

Parser::Parser(Lexer& lexer)
    : lexer(lexer)
    , current(lexer.nextToken())
    , previous(current)
{}

StmtPtr Parser::makeStmt(StmtKind kind, Position pos) { return std::make_unique<Stmt>(std::move(kind), pos); }

ExprPtr Parser::makeExpr(ExprKind kind, Position pos) { return std::make_unique<Expr>(std::move(kind), pos); }

void Parser::advance()
{
    previous = current;
    current  = lexer.nextToken();
}

bool Parser::check(TokenKind k) const { return current.kind == k; }

bool Parser::match(TokenKind k)
{
    if (check(k)) { advance(); return true; }
    return false;
}

void Parser::expect(TokenKind k, const std::string& msg)
{
    if (!check(k)) throw error(msg);
    advance();
}

void Parser::consumeOptionalSemicolon() { match(TokenKind::SEMICOLON); }

ParseError Parser::error(const std::string& msg) const
{
    std::ostringstream oss;
    oss << "[" << current.pos.line << ":" << current.pos.column << "] "
        << msg << " (got '" << current.value << "')";
    return ParseError(oss.str(), current.pos);
}

Program Parser::parse()
{
    Program program;
    while (!check(TokenKind::EOF_TOKEN))
        program.addStmt(parseStatement());
    return program;
}

StmtPtr Parser::parseStatement()
{
    switch (current.kind)
    {
        case TokenKind::KW_LET:    return parseLet();
        case TokenKind::KW_ASYNC:
        case TokenKind::KW_FN:     return parseFnDecl();
        case TokenKind::KW_RETURN: return parseReturn();
        case TokenKind::KW_IF:     return parseIf();
        case TokenKind::KW_FOR:    return parseFor();
        case TokenKind::KW_WHILE:  return parseWhile();
        case TokenKind::KW_MATCH:  return parseMatch();
        case TokenKind::KW_IMPORT: return parseImport();
        case TokenKind::KW_SPAWN:  return parseSpawn();
        case TokenKind::KW_BREAK:    return parseBreak();
        case TokenKind::KW_CONTINUE: return parseContinue();
        case TokenKind::KW_STRUCT:   return parseStruct();
        default:
        {
            // expression statement
            Position p = current.pos;
            StmtKind k;
            k.tag    = StmtKind::Tag::Expr;
            k.expr   = parseExpr();
            consumeOptionalSemicolon();
            return makeStmt(std::move(k), p);
        }
    }
}

Compound Parser::parseCompound()
{
    Compound compound;
    compound.pos = current.pos;
    expect(TokenKind::LBRACE, "expected '{'");
    while (!check(TokenKind::RBRACE) && !check(TokenKind::EOF_TOKEN))
    {
        // track index of return/break for type-checking later
        if (check(TokenKind::KW_RETURN) || check(TokenKind::KW_BREAK))
        {
            if (!compound.break_idx.has_value())
                compound.setBreakIdx(compound.stmts.size());
        }

        // detect implicit return: a non-statement-starting token followed
        // eventually by RBRACE with no SEMICOLON
        bool is_stmt_start =
            check(TokenKind::KW_LET)    ||
            check(TokenKind::KW_FN)     ||
            check(TokenKind::KW_RETURN) ||
            check(TokenKind::KW_IF)     ||
            check(TokenKind::KW_FOR)    ||
            check(TokenKind::KW_WHILE)  ||
            check(TokenKind::KW_MATCH)  ||
            check(TokenKind::KW_IMPORT) ||
            check(TokenKind::KW_SPAWN)  ||
            check(TokenKind::KW_BREAK)  ||
            check(TokenKind::KW_CONTINUE) ||
            check(TokenKind::KW_STRUCT);
        if (!is_stmt_start)
        {
            ExprPtr e = parseExpr();
            // if nothing follows before } it's the tail (implicit return)
            if (check(TokenKind::RBRACE))
            {
                compound.tail_expr = std::move(e);
                break;
            }
            consumeOptionalSemicolon();
            StmtKind k;
            k.tag   = StmtKind::Tag::Expr;
            k.expr  = std::move(e);
            compound.addStmt(makeStmt(std::move(k), previous.pos));
        }
        else
        {
            compound.addStmt(parseStatement());
        }
    }

    expect(TokenKind::RBRACE, "expected '}'");
    return compound;
}

StmtPtr Parser::parseLet()
{
    Position p = current.pos;
    expect(TokenKind::KW_LET, "expected 'let'");
    if (!check(TokenKind::IDENTIFIER))
        throw error("expected identifier after 'let'");
    std::string name = current.value;
    advance();
    // optional type annotation  : T
    TyKind ty = TyKind::Infer;
    std::string user_type_name;
    if (match(TokenKind::COLON))
        ty = parseType();
        if (ty == TyKind::UserDef) user_type_name = previous.value;
    expect(TokenKind::ASSIGN, "expected '=' in let statement");
    StmtKind k;
    k.tag       = StmtKind::Tag::Let;
    k.let_ident = Ident::unqual(IdentInfo{ name, ty, IdentCtx::Def, p });
    k.let_expr  = parseExpr();
    consumeOptionalSemicolon();
    return makeStmt(std::move(k), p);
}

StmtPtr Parser::parseFnDecl()
{
    Position p = current.pos;
    bool is_async = false;
    if (match(TokenKind::KW_ASYNC)) {
        is_async = true;
    }
    expect(TokenKind::KW_FN, "expected 'fn'");
    if (!check(TokenKind::IDENTIFIER))
        throw error("expected function name after 'fn'");
    std::string           fn_name = current.value;
    Position              name_pos = current.pos;
    advance();
    // optional generic names  #(T, U)
    std::vector<std::string> generic_names;
    if (check(TokenKind::HASH))
        generic_names = parseGenericNames();
    active_generic_names.clear();
    for (auto& gn : generic_names) active_generic_names.insert(gn);
    expect(TokenKind::LPAREN, "expected '(' after function name");
    std::vector<Ident> params = parseParamList();
    expect(TokenKind::RPAREN, "expected ')' after parameters");
    TyKind ret_ty = TyKind::Void;
    std::string ret_utn;
    if (match(TokenKind::ARROW))
        ret_ty = parseType();
        if (ret_ty == TyKind::Generic || ret_ty == TyKind::UserDef)
            ret_utn = previous.value;
    // body
    Compound body = parseCompound();
    active_generic_names.clear();
    // build the Func object — ident carries the return type in ty_kind
    Ident fn_ident = Ident::unqual(IdentInfo{ fn_name, ret_ty, IdentCtx::FuncDef, name_pos, ret_utn });
    Func  func(std::move(fn_ident), std::move(params), std::move(body));
    func.generic_names = std::move(generic_names);
    func.is_async = is_async;
    StmtKind k;
    k.tag  = StmtKind::Tag::Func;
    k.func = std::move(func);
    consumeOptionalSemicolon();
    return makeStmt(std::move(k), p);
}


StmtPtr Parser::parseReturn()
{
    Position p = current.pos;
    expect(TokenKind::KW_RETURN, "expected 'return'");
    StmtKind k;
    k.tag = StmtKind::Tag::Return;
    // bare return: next token closes a block, is EOF, or is a semicolon
    if (!check(TokenKind::RBRACE)    &&
        !check(TokenKind::SEMICOLON) &&
        !check(TokenKind::EOF_TOKEN))
    {
        k.ret_expr = parseExpr();
    }
    consumeOptionalSemicolon();
    return makeStmt(std::move(k), p);
}

StmtPtr Parser::parseIf()
{
    Position p = current.pos;
    expect(TokenKind::KW_IF, "expected 'if'");
    StmtKind k;
    k.tag     = StmtKind::Tag::If;
    k.if_cond = parseExpr();
    k.if_body = parseCompound();
    if (match(TokenKind::KW_ELSE))
    {
        if (check(TokenKind::KW_IF))
        {
            // else if — store as chained If stmt
            k.else_or_else_if = parseIf();
        }
        else
        {
            // else — store as Else stmt
            Position else_pos = previous.pos;
            StmtKind else_k;
            else_k.tag       = StmtKind::Tag::Else;
            else_k.else_body = parseCompound();
            k.else_or_else_if = makeStmt(std::move(else_k), else_pos);
        }
    }
    consumeOptionalSemicolon();
    return makeStmt(std::move(k), p);
}

StmtPtr Parser::parseFor()
{
    Position p = current.pos;
    expect(TokenKind::KW_FOR, "expected 'for'");
    if (!check(TokenKind::IDENTIFIER))
        throw error("expected loop variable after 'for'");
    StmtKind k;
    k.tag     = StmtKind::Tag::For;
    k.for_var = current.value;
    advance();
    // 'in' keyword — using KW_FROM as a placeholder until KW_IN is added
    // TODO: add KW_IN to TokenKind and KEYWORDS table in Lexer.cpp
    if (!check(TokenKind::KW_FROM))
        throw error("expected 'in' after loop variable");
    advance();
    k.for_iter = parseExpr();
    k.for_body = parseCompound();
    consumeOptionalSemicolon();
    return makeStmt(std::move(k), p);
}

StmtPtr Parser::parseWhile()
{
    Position p = current.pos;
    expect(TokenKind::KW_WHILE, "expected 'while'");
    StmtKind k;
    k.tag = StmtKind::Tag::While;
    // condition is optional — bare 'while { }' loops forever
    if (!check(TokenKind::LBRACE))
        k.while_cond = parseExpr();
    k.while_body = parseCompound();
    consumeOptionalSemicolon();
    return makeStmt(std::move(k), p);
}

StmtPtr Parser::parseMatch()
{
    Position p = current.pos;
    expect(TokenKind::KW_MATCH, "expected 'match'");
    StmtKind k;
    k.tag           = StmtKind::Tag::Match;
    k.match_subject = parseExpr();
    expect(TokenKind::LBRACE, "expected '{' after match subject");
    while (!check(TokenKind::RBRACE) && !check(TokenKind::EOF_TOKEN))
    {
        k.match_arms.push_back(parseMatchArm());
        consumeOptionalSemicolon();
    }
    expect(TokenKind::RBRACE, "expected '}' to close match");
    consumeOptionalSemicolon();
    return makeStmt(std::move(k), p);
}

StmtPtr Parser::parseImport()
{
    Position p = current.pos;
    expect(TokenKind::KW_IMPORT, "expected 'import'");
    StmtKind k;
    k.tag         = StmtKind::Tag::Import;
    if (check(TokenKind::STRING) || check(TokenKind::IDENTIFIER))
    {
        k.import_path = current.value;
        advance();
    }
    else 
    {
        throw error("expected string path or identifier after 'import'");
    }
    if (match(TokenKind::KW_AS))
    {
        if (!check(TokenKind::IDENTIFIER))
            throw error("expected alias name after 'as'");
        k.import_alias = current.value;
        advance();
    }
    consumeOptionalSemicolon();
    return makeStmt(std::move(k), p);
}

StmtPtr Parser::parseSpawn()
{
    Position p = current.pos;
    expect(TokenKind::KW_SPAWN, "expected 'spawn'");
    if (!check(TokenKind::KW_FN))
        throw error("expected 'fn' after 'spawn'");
    StmtKind k;
    k.tag      = StmtKind::Tag::Spawn;
    k.spawn_fn = parseFnDecl();
    consumeOptionalSemicolon();
    return makeStmt(std::move(k), p);
}

StmtPtr Parser::parseBreak()
{
    Position p = current.pos;
    expect(TokenKind::KW_BREAK, "expected 'break'");
    StmtKind k;
    k.tag = StmtKind::Tag::Break;
    consumeOptionalSemicolon();
    return makeStmt(std::move(k), p);
}

StmtPtr Parser::parseContinue()
{
    Position p = current.pos;
    expect(TokenKind::KW_CONTINUE, "expected 'continue'");
    StmtKind k;
    k.tag = StmtKind::Tag::Continue;
    consumeOptionalSemicolon();
    return makeStmt(std::move(k), p);
}

StmtPtr Parser::parseStruct()
{
    Position p = current.pos;
    expect(TokenKind::KW_STRUCT, "expected 'struct'");
    if (!check(TokenKind::IDENTIFIER))
        throw error("expected struct name after 'struct'");
    std::string struct_name = current.value;
    advance();
    expect(TokenKind::LBRACE, "expected '{' after struct name");
    std::vector<StructField> fields;
    while (!check(TokenKind::RBRACE) && !check(TokenKind::EOF_TOKEN))
    {
        if (!check(TokenKind::IDENTIFIER))
            throw error("expected field name in struct");
        std::string field_name = current.value;
        Position    field_pos  = current.pos;
        advance();
        expect(TokenKind::COLON, "expected ':' after field name");
        TyKind field_ty = parseType();
        fields.emplace_back(std::move(field_name), field_ty, field_pos);
        match(TokenKind::COMMA);
    }
    expect(TokenKind::RBRACE, "expected '}' to close struct");
    consumeOptionalSemicolon();
    return makeStmt(StmtKind::makeStruct(std::move(struct_name), std::move(fields)), p);
}

ExprPtr Parser::parseExpr()   { return parsePipe(); }

ExprPtr Parser::parsePipe()
{
    ExprPtr lhs = parseAssign();
    while (check(TokenKind::PIPE))
    {
        Position p = current.pos;
        advance();
        ExprPtr rhs = parseAssign();
        ExprKind k;
        k.tag      = ExprKind::Tag::Pipe;
        k.pipe_lhs = std::move(lhs);
        k.pipe_rhs = std::move(rhs);
        lhs        = makeExpr(std::move(k), p);
    }
    return lhs;
}

ExprPtr Parser::parseAssign()
{
    ExprPtr lhs = parseOr();
    BinOp op;
    bool  is_assign = false;
    if      (check(TokenKind::ASSIGN))       { op = BinOp::Assign;    is_assign = true; }
    else if (check(TokenKind::PLUS_ASSIGN))  { op = BinOp::AddAssign; is_assign = true; }
    else if (check(TokenKind::MINUS_ASSIGN)) { op = BinOp::SubAssign; is_assign = true; }
    else if (check(TokenKind::STAR_ASSIGN))  { op = BinOp::MulAssign; is_assign = true; }
    else if (check(TokenKind::SLASH_ASSIGN)) { op = BinOp::DivAssign; is_assign = true; }
    if (is_assign)
    {
        Position p = current.pos;
        advance();
        ExprPtr rhs = parseAssign(); // right-associative

        ExprKind k;
        k.tag    = ExprKind::Tag::Assign;
        k.bin_op = op;
        k.lhs    = std::move(lhs);
        k.rhs    = std::move(rhs);
        return makeExpr(std::move(k), p);
    }
    return lhs;
}

ExprPtr Parser::parseOr()
{
    ExprPtr lhs = parseAnd();
    while (check(TokenKind::OR))
    {
        Position p = current.pos;
        advance();
        ExprPtr rhs = parseAnd();
        ExprKind k;
        k.tag    = ExprKind::Tag::Binary;
        k.bin_op = BinOp::Or;
        k.lhs    = std::move(lhs);
        k.rhs    = std::move(rhs);
        lhs      = makeExpr(std::move(k), p);
    }
    return lhs;
}

ExprPtr Parser::parseAnd()
{
    ExprPtr lhs = parseEquality();
    while (check(TokenKind::AND))
    {
        Position p = current.pos;
        advance();
        ExprPtr rhs = parseEquality();
        ExprKind k;
        k.tag    = ExprKind::Tag::Binary;
        k.bin_op = BinOp::And;
        k.lhs    = std::move(lhs);
        k.rhs    = std::move(rhs);
        lhs      = makeExpr(std::move(k), p);
    }
    return lhs;
}

ExprPtr Parser::parseEquality()
{
    ExprPtr lhs = parseComparison();
    while (check(TokenKind::EQ) || check(TokenKind::NEQ))
    {
        Position p  = current.pos;
        BinOp    op = check(TokenKind::EQ) ? BinOp::Eq : BinOp::Neq;
        advance();
        ExprPtr rhs = parseComparison();
        ExprKind k;
        k.tag    = ExprKind::Tag::Binary;
        k.bin_op = op;
        k.lhs    = std::move(lhs);
        k.rhs    = std::move(rhs);
        lhs      = makeExpr(std::move(k), p);
    }
    return lhs;
}

ExprPtr Parser::parseComparison()
{
    ExprPtr lhs = parseAdditive();
    while (check(TokenKind::LT)  || check(TokenKind::GT) ||
           check(TokenKind::LTE) || check(TokenKind::GTE))
    {
        Position p = current.pos;
        BinOp op;
        switch (current.kind)
        {
            case TokenKind::LT:  op = BinOp::Lt;  break;
            case TokenKind::GT:  op = BinOp::Gt;  break;
            case TokenKind::LTE: op = BinOp::Lte; break;
            default:             op = BinOp::Gte; break;
        }
        advance();
        ExprPtr rhs = parseAdditive();
        ExprKind k;
        k.tag    = ExprKind::Tag::Binary;
        k.bin_op = op;
        k.lhs    = std::move(lhs);
        k.rhs    = std::move(rhs);
        lhs      = makeExpr(std::move(k), p);
    }
    return lhs;
}

ExprPtr Parser::parseAdditive()
{
    ExprPtr lhs = parseMultiplicative();
    while (check(TokenKind::PLUS) || check(TokenKind::MINUS))
    {
        Position p  = current.pos;
        BinOp    op = check(TokenKind::PLUS) ? BinOp::Add : BinOp::Sub;
        advance();
        ExprPtr rhs = parseMultiplicative();
        ExprKind k;
        k.tag    = ExprKind::Tag::Binary;
        k.bin_op = op;
        k.lhs    = std::move(lhs);
        k.rhs    = std::move(rhs);
        lhs      = makeExpr(std::move(k), p);
    }
    return lhs;
}

ExprPtr Parser::parseMultiplicative()
{
    ExprPtr lhs = parseMatMul();
    while (check(TokenKind::STAR) || check(TokenKind::SLASH))
    {
        Position p  = current.pos;
        BinOp    op = check(TokenKind::STAR) ? BinOp::Mul : BinOp::Div;
        advance();
        ExprPtr rhs = parseMatMul();
        ExprKind k;
        k.tag    = ExprKind::Tag::Binary;
        k.bin_op = op;
        k.lhs    = std::move(lhs);
        k.rhs    = std::move(rhs);
        lhs      = makeExpr(std::move(k), p);
    }
    return lhs;
}

ExprPtr Parser::parseMatMul()
{
    ExprPtr lhs = parseUnary();
    while (check(TokenKind::MATMUL))
    {
        Position p = current.pos;
        advance();
        ExprPtr rhs = parseUnary();
        ExprKind k;
        k.tag    = ExprKind::Tag::Binary;
        k.bin_op = BinOp::MatMul;
        k.lhs    = std::move(lhs);
        k.rhs    = std::move(rhs);
        lhs      = makeExpr(std::move(k), p);
    }
    return lhs;
}

ExprPtr Parser::parseUnary()
{
    if (check(TokenKind::NOT) || check(TokenKind::MINUS))
    {
        Position p  = current.pos;
        UnaryOp  op = check(TokenKind::NOT) ? UnaryOp::Not : UnaryOp::Neg;
        advance();
        ExprKind k;
        k.tag      = ExprKind::Tag::Unary;
        k.unary_op = op;
        k.operand  = parseUnary();
        return makeExpr(std::move(k), p);
    }
    return parsePostfix();
}

ExprPtr Parser::parsePostfix()
{
    ExprPtr expr = parsePrimary();
    while (true)
    {
        if (check(TokenKind::LPAREN))
        {
            Position p = current.pos;
            advance();
            std::vector<ExprPtr> args = parseExprList(TokenKind::RPAREN);
            expect(TokenKind::RPAREN, "expected ')' after arguments");
            ExprKind k;
            k.tag    = ExprKind::Tag::Call;
            k.callee = std::move(expr);
            k.args   = std::move(args);
            expr     = makeExpr(std::move(k), p);
        }
        else if (check(TokenKind::LBRACKET))
        {
            Position p = current.pos;
            advance();
            ExprPtr idx = parseExpr();
            expect(TokenKind::RBRACKET, "expected ']' after index");
            ExprKind k;
            k.tag   = ExprKind::Tag::Index;
            k.target = std::move(expr);
            k.index = std::move(idx);
            expr    = makeExpr(std::move(k), p);
        }
        else if (check(TokenKind::DOT))
        {
            Position p = current.pos;
            advance();
            if (!check(TokenKind::IDENTIFIER))
                throw error("expected field name after '.'");
            std::string mem = current.value;
            advance();
            ExprKind k;
            k.tag    = ExprKind::Tag::Field;
            k.target = std::move(expr);
            k.member = mem;
            expr     = makeExpr(std::move(k), p);
        }
        else if (check(TokenKind::DOUBLE_COLON))
        {
            Position p = current.pos;
            advance();
            if (!check(TokenKind::IDENTIFIER))
                throw error("expected name after '::'");
            std::string mem = current.value;
            advance();
            ExprKind k;
            k.tag    = ExprKind::Tag::Scope;
            k.target = std::move(expr);
            k.member = mem;
            expr     = makeExpr(std::move(k), p);
        }
        else if (check(TokenKind::CHANNEL_SEND))
        {
            expr = parseChannelSend(std::move(expr));
        }
        else if (check(TokenKind::LBRACE)) 
        {
            if (expr->kind.tag == ExprKind::Tag::Id)
            {
                std::string struct_name = expr->kind.id.info.name;
                expr = parseStructLit(struct_name, expr->pos);
            }
            else
            {
                break; 
            }
        }
        else
        {
            break;
        }
    }
    return expr;
}

ExprPtr Parser::parsePrimary()
{
    switch (current.kind)
    {
        case TokenKind::INT:        return parseIntLit();
        case TokenKind::FLOAT:      return parseFloatLit();
        case TokenKind::STRING:     return parseStringLit();
        case TokenKind::KW_TRUE:
        case TokenKind::KW_FALSE:   return parseBoolLit();
        case TokenKind::IDENTIFIER: return parseIdentOrCall();
        case TokenKind::LPAREN:     return parseGrouped();
        case TokenKind::LBRACKET:   return parseArrayLit();
        case TokenKind::KW_TENSOR:  return parseTensorLit();
        case TokenKind::KW_SET:     return parseSetLit();
        case TokenKind::KW_MAP:     return parseMapLit();
        case TokenKind::KW_QUEUE:   return parseQueueLit();
        case TokenKind::KW_STACK:   return parseStackLit();
        case TokenKind::KW_TUPLE:   return parseTupleLit();
        case TokenKind::KW_IF:      return parseIfExpr();
        case TokenKind::KW_MATCH:   return parseMatchExpr();
        case TokenKind::LBRACE:     return parseBlockExpr();
        case TokenKind::KW_FN:      return parseFnExpr();
        case TokenKind::KW_SPAWN:   return parseSpawnExpr();
        case TokenKind::KW_AWAIT:   return parseAwait();
        case TokenKind::KW_GRAD:    return parseGrad();
        default:
            throw error("unexpected token in expression");
    }
}

// literal helpers — private, not declared in header, called from parsePrimary
ExprPtr Parser::parseIntLit()
{
    ExprKind k;
    k.tag = ExprKind::Tag::Lit;
    k.lit = LitKind::makeInt(current.value);
    Position p = current.pos;
    advance();
    return makeExpr(std::move(k), p);
}

ExprPtr Parser::parseFloatLit()
{
    ExprKind k;
    k.tag = ExprKind::Tag::Lit;
    k.lit = LitKind::makeFloat(current.value);
    Position p = current.pos;
    advance();
    return makeExpr(std::move(k), p);
}

ExprPtr Parser::parseStringLit()
{
    ExprKind k;
    k.tag = ExprKind::Tag::Lit;
    k.lit = LitKind::makeStr(current.value);
    Position p = current.pos;
    advance();
    return makeExpr(std::move(k), p);
}

ExprPtr Parser::parseBoolLit()
{
    ExprKind k;
    k.tag = ExprKind::Tag::Lit;
    k.lit = LitKind::makeBool(current.kind == TokenKind::KW_TRUE);
    Position p = current.pos;
    advance();
    return makeExpr(std::move(k), p);
}

ExprPtr Parser::parseStructLit(const std::string& name, Position p)
{
    expect(TokenKind::LBRACE, "expected '{' for struct literal");
    
    std::vector<std::pair<std::string, ExprPtr>> fields;
    
    while (!check(TokenKind::RBRACE) && !check(TokenKind::EOF_TOKEN))
    {
        if (!check(TokenKind::IDENTIFIER))
            throw error("expected field name in struct literal");
            
        std::string field_name = current.value;
        advance();
        
        expect(TokenKind::COLON, "expected ':' after field name");
        
        ExprPtr val = parseExpr();
        fields.push_back({field_name, std::move(val)});
        
        if (!match(TokenKind::COMMA))
            break;
    }
    
    expect(TokenKind::RBRACE, "expected '}' after struct literal");
    ExprKind k;
    k.tag = ExprKind::Tag::StructLit; // Double check this Tag exists in your AST!
    k.struct_init_name = name;
    k.struct_init_fields = std::move(fields);
    
    return makeExpr(std::move(k), p);
}

ExprPtr Parser::parseIdentOrCall()
{
    Position p = current.pos;
    std::string name = current.value;
    advance();
    ExprKind k;
    k.tag = ExprKind::Tag::Id;
    k.id  = Ident::unqual(IdentInfo{ name, TyKind::Infer, IdentCtx::Ref, p });
    return makeExpr(std::move(k), p);
    // actual call detection happens in parsePostfix()
}

ExprPtr Parser::parseGrouped()
{
    expect(TokenKind::LPAREN, "expected '('");
    ExprPtr e = parseExpr();
    expect(TokenKind::RPAREN, "expected ')'");
    return e;
}

ExprPtr Parser::parseIfExpr()
{
    Position p = current.pos;
    expect(TokenKind::KW_IF, "expected 'if'");
    ExprKind k;
    k.tag        = ExprKind::Tag::If;
    k.condition  = parseExpr();
    k.then_branch = parseBlockExpr();
    if (match(TokenKind::KW_ELSE))
        k.else_branch = check(TokenKind::KW_IF) ? parseIfExpr() : parseBlockExpr();
    return makeExpr(std::move(k), p);
}

ExprPtr Parser::parseMatchExpr()
{
    Position p = current.pos;
    expect(TokenKind::KW_MATCH, "expected 'match'");
    ExprKind k;
    k.tag           = ExprKind::Tag::Match;
    k.match_subject = parseExpr();
    expect(TokenKind::LBRACE, "expected '{' after match subject");
    while (!check(TokenKind::RBRACE) && !check(TokenKind::EOF_TOKEN))
    {
        k.arms.push_back(parseMatchArm());
        consumeOptionalSemicolon();
    }
    expect(TokenKind::RBRACE, "expected '}'");
    return makeExpr(std::move(k), p);
}

MatchArm Parser::parseMatchArm()
{
    MatchArm arm;
    arm.pos     = current.pos;
    arm.pattern = parseExpr();
    if (match(TokenKind::KW_IF))
        arm.guard = parseExpr();
    expect(TokenKind::FAT_ARROW, "expected '=>' in match arm");
    switch (current.kind)
    {
        case TokenKind::KW_LET:
        case TokenKind::KW_FN:
        case TokenKind::KW_RETURN:
        case TokenKind::KW_IF:
        case TokenKind::KW_FOR:
        case TokenKind::KW_WHILE:
        case TokenKind::KW_MATCH:
        case TokenKind::KW_IMPORT:
        case TokenKind::KW_SPAWN:
            arm.body_stmt = parseStatement();
            break;
        default:
            arm.body = parseExpr();
            break;
    }
    return arm;
}

ExprPtr Parser::parseBlockExpr()
{
    Position p = current.pos;
    ExprKind k;
    k.tag   = ExprKind::Tag::Block;
    k.block = parseCompound();
    return makeExpr(std::move(k), p);
}


ExprPtr Parser::parseFnExpr()
{
    Position p = current.pos;
    expect(TokenKind::KW_FN, "expected 'fn'");
    ExprKind k;
    k.tag = ExprKind::Tag::FnExpr;
    if (check(TokenKind::HASH))
        k.fn_generic_names = parseGenericNames();
    expect(TokenKind::LPAREN, "expected '('");
    for (const Ident& ident : parseParamList())
        k.fn_params.push_back({ ident.name(), ident.ty_kind() });
    expect(TokenKind::RPAREN, "expected ')'");
    k.fn_ret_type = TyKind::Void;
    if (match(TokenKind::ARROW))
        k.fn_ret_type = parseType();
    k.fn_body = parseCompound();

    return makeExpr(std::move(k), p);
}

ExprPtr Parser::parseSpawnExpr()
{
    Position p = current.pos;
    advance(); // consume 'spawn'
    ExprPtr call = parsePostfix(); 
    if (call->kind.tag != ExprKind::Tag::Call) {
        throw error("expected function call after 'spawn'");
    }
    return makeExpr(ExprKind::makeSpawn(std::move(call)), p);
}

ExprPtr Parser::parseAwait()
{
    Position p = current.pos;
    expect(TokenKind::KW_AWAIT, "expected 'await'");
    ExprKind k;
    k.tag     = ExprKind::Tag::Await;
    k.awaited = parseExpr();
    return makeExpr(std::move(k), p);
}

ExprPtr Parser::parseGrad()
{
    Position p = current.pos;
    expect(TokenKind::KW_GRAD,   "expected 'grad'");
    expect(TokenKind::LPAREN,    "expected '(' after 'grad'");
    ExprKind k;
    k.tag         = ExprKind::Tag::Grad;
    k.grad_loss   = parseExpr();
    expect(TokenKind::COMMA,     "expected ',' in grad(loss, params)");
    k.grad_params = parseExpr();
    expect(TokenKind::RPAREN,    "expected ')'");
    return makeExpr(std::move(k), p);
}

ExprPtr Parser::parseChannelSend(ExprPtr lhs)
{
    Position p = current.pos;
    expect(TokenKind::CHANNEL_SEND, "expected '<-'");
    ExprKind k;
    k.tag      = ExprKind::Tag::ChannelSend;
    k.channel  = std::move(lhs);
    k.send_val = parseExpr();
    return makeExpr(std::move(k), p);
}

ExprPtr Parser::parseArrayLit()
{
    Position p = current.pos;
    expect(TokenKind::LBRACKET, "expected '['");
    ExprKind k;
    k.tag      = ExprKind::Tag::ArrayLit;
    k.elements = parseExprList(TokenKind::RBRACKET);
    expect(TokenKind::RBRACKET, "expected ']'");
    return makeExpr(std::move(k), p);
}

ExprPtr Parser::parseTensorLit()
{
    Position p = current.pos;
    expect(TokenKind::KW_TENSOR, "expected 'Tensor'");
    ExprKind k;
    k.tag = ExprKind::Tag::TensorLit;
    if (check(TokenKind::HASH))
        k.generic_params = parseGenericParams();
    expect(TokenKind::LBRACKET, "expected '[' for Tensor literal");
    k.rows = parseTensorRows();
    expect(TokenKind::RBRACKET, "expected ']'");
    return makeExpr(std::move(k), p);
}

ExprPtr Parser::parseSetLit()
{
    Position p = current.pos;
    expect(TokenKind::KW_SET, "expected 'Set'");
    expect(TokenKind::LBRACKET, "expected '['");
    ExprKind k;
    k.tag      = ExprKind::Tag::SetLit;
    k.elements = parseExprList(TokenKind::RBRACKET);
    expect(TokenKind::RBRACKET, "expected ']'");
    return makeExpr(std::move(k), p);
}

ExprPtr Parser::parseMapLit()
{
    Position p = current.pos;
    expect(TokenKind::KW_MAP,  "expected 'Map'");
    expect(TokenKind::LBRACE,  "expected '{'");
    ExprKind k;
    k.tag = ExprKind::Tag::MapLit;
    while (!check(TokenKind::RBRACE) && !check(TokenKind::EOF_TOKEN))
    {
        ExprPtr key = parseExpr();
        expect(TokenKind::COLON, "expected ':' in Map entry");
        ExprPtr val = parseExpr();
        k.map_pairs.push_back({ std::move(key), std::move(val) });
        match(TokenKind::COMMA);
    }
    expect(TokenKind::RBRACE, "expected '}'");
    return makeExpr(std::move(k), p);
}

ExprPtr Parser::parseQueueLit()
{
    Position p = current.pos;
    expect(TokenKind::KW_QUEUE, "expected 'Queue'");
    expect(TokenKind::LBRACKET, "expected '['");
    ExprKind k;
    k.tag      = ExprKind::Tag::QueueLit;
    k.elements = parseExprList(TokenKind::RBRACKET);
    expect(TokenKind::RBRACKET, "expected ']'");
    return makeExpr(std::move(k), p);
}

ExprPtr Parser::parseStackLit()
{
    Position p = current.pos;
    expect(TokenKind::KW_STACK, "expected 'Stack'");
    expect(TokenKind::LBRACKET, "expected '['");
    ExprKind k;
    k.tag      = ExprKind::Tag::StackLit;
    k.elements = parseExprList(TokenKind::RBRACKET);
    expect(TokenKind::RBRACKET, "expected ']'");
    return makeExpr(std::move(k), p);
}

ExprPtr Parser::parseTupleLit()
{
    Position p = current.pos;
    expect(TokenKind::KW_TUPLE, "expected 'Tuple'");
    expect(TokenKind::LBRACKET, "expected '['");
    ExprKind k;
    k.tag      = ExprKind::Tag::TupleLit;
    k.elements = parseExprList(TokenKind::RBRACKET);
    expect(TokenKind::RBRACKET, "expected ']'");
    return makeExpr(std::move(k), p);
}

std::vector<std::vector<ExprPtr>> Parser::parseTensorRows()
{
    std::vector<std::vector<ExprPtr>> rows;
    std::vector<ExprPtr> row;
    while (!check(TokenKind::RBRACKET) && !check(TokenKind::EOF_TOKEN))
    {
        row.push_back(parseExpr());
        if (match(TokenKind::COMMA))
            continue;
        if (check(TokenKind::ROW_SEP))
        {
            advance();
            rows.push_back(std::move(row));
            row.clear();
        }
    }
    if (!row.empty())
        rows.push_back(std::move(row));
    return rows;
}


TyKind Parser::parseType()
{
    switch (current.kind)
    {
        case TokenKind::KW_I32:    advance(); return TyKind::I32;
        case TokenKind::KW_I64:    advance(); return TyKind::I64;
        case TokenKind::KW_F32:    advance(); return TyKind::F32;
        case TokenKind::KW_F64:    advance(); return TyKind::F64;
        case TokenKind::KW_BOOL:   advance(); return TyKind::Bool;
        case TokenKind::KW_STR:    advance(); return TyKind::Str;
        case TokenKind::KW_VOID:   advance(); return TyKind::Void;
        case TokenKind::KW_TENSOR: advance(); return TyKind::Tensor;
        case TokenKind::KW_MAP:    advance(); return TyKind::Map;
        case TokenKind::KW_SET:    advance(); return TyKind::Set;
        case TokenKind::KW_QUEUE:  advance(); return TyKind::Queue;
        case TokenKind::KW_STACK:  advance(); return TyKind::Stack;
        case TokenKind::KW_TUPLE:  advance(); return TyKind::Tuple;
        case TokenKind::KW_FN:     advance(); return TyKind::FnType;
        case TokenKind::IDENTIFIER:
        {
            std::string id = current.value;
            advance();
            bool is_generic = active_generic_names.count(id) > 0;
            return is_generic ? TyKind::Generic : TyKind::UserDef;
        }
        default:
            throw error("expected type annotation");
    }
}

GenericParams Parser::parseGenericParams()
{
    GenericParams gp;
    gp.pos = current.pos;
    expect(TokenKind::HASH,   "expected '#'");
    expect(TokenKind::LPAREN, "expected '(' after '#'");
    while (!check(TokenKind::RPAREN) && !check(TokenKind::EOF_TOKEN))
    {
        if (check(TokenKind::LBRACKET))
        {
            advance(); // consume [
            while (!check(TokenKind::RBRACKET) && !check(TokenKind::EOF_TOKEN))
            {
                if (check(TokenKind::INT))
                {
                    gp.shape.push_back(std::stoi(current.value));
                    advance();
                }
                match(TokenKind::COMMA);
            }
            expect(TokenKind::RBRACKET, "expected ']' after shape");
        }
        else
        {
            gp.type_params.push_back(parseType());
        }
        match(TokenKind::COMMA);
    }
    expect(TokenKind::RPAREN, "expected ')' after generic params");
    return gp;
}

std::vector<std::string> Parser::parseGenericNames()
{
    std::vector<std::string> names;
    expect(TokenKind::HASH,   "expected '#'");
    expect(TokenKind::LPAREN, "expected '(' after '#'");
    while (!check(TokenKind::RPAREN) && !check(TokenKind::EOF_TOKEN))
    {
        if (!check(TokenKind::IDENTIFIER))
            throw error("expected generic type name");
        names.push_back(current.value);
        advance();
        match(TokenKind::COMMA);
    }
    expect(TokenKind::RPAREN, "expected ')' after generic names");
    return names;
}

std::vector<Ident> Parser::parseParamList()
{
    std::vector<Ident> params;
    while (!check(TokenKind::RPAREN) && !check(TokenKind::EOF_TOKEN))
    {
        if (!check(TokenKind::IDENTIFIER))
            throw error("expected parameter name");
        std::string name = current.value;
        Position    p    = current.pos;
        advance();
        expect(TokenKind::COLON, "expected ':' after parameter name");
        TyKind ty = parseType();
        std::string utn;
        if (ty == TyKind::Generic || ty == TyKind::UserDef)
            utn = previous.value;
        params.push_back(Ident::unqual(IdentInfo{ name, ty, IdentCtx::Param, p, utn }));
        match(TokenKind::COMMA);
    }
    return params;
}

std::vector<ExprPtr> Parser::parseExprList(TokenKind terminator)
{
    std::vector<ExprPtr> list;
    while (!check(terminator) && !check(TokenKind::EOF_TOKEN))
    {
        list.push_back(parseExpr());
        match(TokenKind::COMMA);
    }
    return list;
}