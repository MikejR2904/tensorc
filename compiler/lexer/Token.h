#pragma once

#include <string>
#include "TokenKind.h"
#include "Position.h"

// Each token produced has a kind (what type of token it is), a string value (the actual value), and a position in the source code
struct Token
{
    TokenKind kind;
    std::string value;
    Position pos;

    Token(TokenKind k, const std::string& v, Position p) : kind(k), value(v), pos(p) {}
};
