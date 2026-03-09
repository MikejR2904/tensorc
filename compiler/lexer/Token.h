#pragma once

#include <string>
#include "TokenKind.h"
#include "Position.h"

struct Token
{
    TokenKind kind;
    std::string value;
    Position pos;

    Token(TokenKind k, const std::string& v, Position p)
        : kind(k), value(v), pos(p) {}
};