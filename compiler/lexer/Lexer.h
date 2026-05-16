#pragma once

#include <string>
#include <vector>
#include "Token.h"
#include "Position.h"

// Convert source code string (.tcc) into a stream of Tokens for the parser 
enum class LexContext
{
    DEFAULT,
    ARRAY,   // inside plain [...]
    TENSOR,  // inside Tensor[...] where ';' becomes ROW_SEP
    SET,     // Set[...]
    MAP,     // Map{...}
    QUEUE,   // Queue[...]
    STACK,   // Stack[...]
    TUPLE,   // Tuple[...]
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
    char current; // which character is currently being looked at
    Position pos;

    std::vector<LexContext> context_stack; // to store in which context currently is, for context-sensitive lexing
    TokenKind last_token_kind; // to determine context for next token (e.g. '{' after 'Map' vs. plain block)

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