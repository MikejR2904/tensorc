#pragma once

#include <string>
#include <vector>
#include "Token.h"
#include "Position.h"

enum class LexContext
{
    DEFAULT,
    ARRAY,    // inside plain [...]
    TENSOR,   // inside Tensor[...]  — ';' becomes ROW_SEP
    SET,      // Set[...]
    MAP,      // Map{...}
    QUEUE,    // Qeueue[...]
    STACK,    // Stack[...]
    TUPLE,    // Tuple[...]
};

class Lexer
{
public:
    explicit Lexer(const std::string& src);
    Token nextToken();
    Token peekToken();
    bool eof() const;

private:
    std::string source;
    size_t index;
    char current;
    Position pos;

    std::vector<LexContext> context_stack;
    TokenKind               last_token_kind;

    void nextChar();
    char peekChar() const;

    void skipWhitespace();
    void skipComment();

    Token lexIdentifier();
    Token lexNumber();
    Token lexString();
    Token lexOperator();

    LexContext currentContext() const;
    void pushContext(LexContext ctx);
    void popContext();

    Token makeToken(TokenKind kind, const std::string& value, Position start);
};