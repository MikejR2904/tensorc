#pragma once
#include <string>

enum class TokenKind
{
    EOF_TOKEN,

    IDENTIFIER,
    NUMBER,
    STRING,
    FLOAT,
    INT,

    // control flow keywords
    KW_IF,
    KW_ELSE,
    KW_FOR,
    KW_WHILE,
    KW_BREAK,
    KW_CONTINUE,
    KW_MATCH,
    KW_RETURN,

    // declaration keywords
    KW_FN,
    KW_LET,
    KW_PUB,

    // primitive type keywords
    KW_I32, KW_I64,
    KW_F32, KW_F64,
    KW_BOOL,
    KW_TRUE, KW_FALSE, // boolean literals
    KW_STR,
    KW_VOID,

    // collection type keywords
    KW_MAP,      // Map{...}
    KW_SET,      // Set[...]
    KW_TUPLE,    // Tuple[...]
    KW_QUEUE,    // Queue[...] for distributed pipelines
    KW_STACK,    // Stack[...]
    KW_TENSOR,         // also triggers context-switch for '[' and ';'
    KW_GRAD,           // gradient context

    // async or distributed keywords
    KW_ASYNC,          // distributed async ops
    KW_AWAIT,
    KW_SPAWN,          // spawn a distributed worker
    KW_YIELD,          // generator / lazy evaluation

    // module system keywords
    KW_IMPORT,
    KW_FROM,
    KW_AS,

    // arithmetic operators
    PLUS,             // +
    MINUS,            // -
    STAR,             // *
    SLASH,            // /
    MATMUL,           // @ matrix multiply

    // augmented assignment
    PLUS_ASSIGN,      // +=
    MINUS_ASSIGN,     // -=
    STAR_ASSIGN,      // *=
    SLASH_ASSIGN,     // /=

    // comparison operators
    ASSIGN,          // =
    EQ,              // ==
    NEQ,             // !=
    LT,              // <
    GT,              // >
    LTE,             // <=
    GTE,             // >=

    // logical operators
    AND,             // &&
    OR,              // ||
    NOT,             // !

    // arrow / pipe
    ARROW,           // ->   return type annotation
    FAT_ARROW,       // =>   match arm / lambda
    PIPE,            // |>   pipeline operator (e.g., model |> train |> evaluate)

    // distributed / channel
    CHANNEL_SEND,    // <-   send to channel

    // generic sigil
    HASH,            // #    e.g. Tensor#(f32, [3,4])

    // scope / access
    DOUBLE_COLON,    // ::
    DOT,             // .
    RANGE,           // ..
    ELLIPSIS,        // ...

    // delimiters
    LPAREN,         // (
    RPAREN,         // )
    LBRACE,         // {
    RBRACE,         // }
    LBRACKET,       // [
    RBRACKET,       // ]

    // punctuation
    COMMA,          // ,
    COLON,          // :
    SEMICOLON,      // ;
    ROW_SEP,        // ;  inside Tensor[...]  — row separator

    UNKNOWN         // as fallback
};