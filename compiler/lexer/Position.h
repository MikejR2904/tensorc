#pragma once

struct Position
{
    int line; // indicate the line number in the source code (starting from 1)
    int column; // indicate the column number in the source code (starting from 1)
    // line x and column y mean the y-th character in the x-th line of the source code
    Position(int l=1, int c=1) : line(l), column(c) {}
};