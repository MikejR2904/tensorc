#pragma once

struct Position
{
    int line;
    int column;

    Position(int l=1, int c=1)
        : line(l), column(c) {}
};