#include "Lexer.h"
#include <cctype>
#include <unordered_map>
#include <unordered_set>

static const std::unordered_map<std::string, TokenKind> KEYWORDS = {
    // control flow
    { "if",       TokenKind::KW_IF       },
    { "else",     TokenKind::KW_ELSE     },
    { "for",      TokenKind::KW_FOR      },
    { "while",    TokenKind::KW_WHILE    },
    { "break",    TokenKind::KW_BREAK    },
    { "continue", TokenKind::KW_CONTINUE },
    { "return",   TokenKind::KW_RETURN   },
    { "match",    TokenKind::KW_MATCH    },
    // declarations
    { "fn",       TokenKind::KW_FN       },
    { "let",      TokenKind::KW_LET      },
    { "pub",      TokenKind::KW_PUB      },
    // primitive types
    { "i32",      TokenKind::KW_I32      },
    { "i64",      TokenKind::KW_I64      },
    { "f32",      TokenKind::KW_F32      },
    { "f64",      TokenKind::KW_F64      },
    { "bool",     TokenKind::KW_BOOL     },
    { "str",      TokenKind::KW_STR      },
    { "void",     TokenKind::KW_VOID     },
    // collection types
    { "Tensor",   TokenKind::KW_TENSOR   },
    { "Map",      TokenKind::KW_MAP      },
    { "Set",      TokenKind::KW_SET      },
    { "Queue",    TokenKind::KW_QUEUE    },
    { "Stack",    TokenKind::KW_STACK    },
    { "Tuple",    TokenKind::KW_TUPLE    },
    { "grad",     TokenKind::KW_GRAD     },
    // async / distributed
    { "async",    TokenKind::KW_ASYNC    },
    { "await",    TokenKind::KW_AWAIT    },
    { "spawn",    TokenKind::KW_SPAWN    },
    { "yield",    TokenKind::KW_YIELD    },
    // module system
    { "import",   TokenKind::KW_IMPORT   },
    { "from",     TokenKind::KW_FROM     },
    { "as",       TokenKind::KW_AS       },
    { "in",       TokenKind::KW_IN       },
    { "struct",   TokenKind::KW_STRUCT   },
    // boolean literals
    { "true",     TokenKind::KW_TRUE     },
    { "false",    TokenKind::KW_FALSE    },
};

static bool isBraceCollectionOpener(TokenKind k)
{
    return k == TokenKind::KW_MAP;
}

static LexContext bracketContextFor(TokenKind k)
{
    switch (k)
    {
        case TokenKind::KW_TENSOR: return LexContext::TENSOR;
        case TokenKind::KW_SET:    return LexContext::SET;
        case TokenKind::KW_QUEUE:  return LexContext::QUEUE;
        case TokenKind::KW_STACK:  return LexContext::STACK;
        case TokenKind::KW_TUPLE:  return LexContext::TUPLE;
        case TokenKind::RPAREN:    return LexContext::TENSOR; // Tensor#(...)[
        default:                   return LexContext::ARRAY;
    }
}

Lexer::Lexer(const std::string& src)
{
    source = src;
    index = 0;
    pos = Position(1,1);
    current = source.empty() ? '\0' : source[0];
    last_token_kind = TokenKind::EOF_TOKEN;
}

void Lexer::nextChar()
{
    if(current == '\0')
        return;
    if(current == '\n')
    {
        pos.line++;
        pos.column = 1;
    }
    else
    {
        pos.column++;
    }
    index++;
    current = (index >= source.size()) ? '\0' : source[index];
}

char Lexer::peekChar() const
{
    size_t next = index + 1;
    return (next >= source.size()) ? '\0' : source[next];
}

bool Lexer::eof() const
{
    return current == '\0';
}

void Lexer::skipWhitespace()
{
    while(std::isspace(static_cast<unsigned char>(current)))
        nextChar();
}

void Lexer::skipComment()
{
    while (true)
    {
        if (current == '/' && peekChar() == '/')      // single-line comment
        {
            while (current != '\n' && !eof())
                nextChar();
        }
        else if (current == '/' && peekChar() == '*') // block comment  /* ... */
        {
            nextChar(); nextChar();
            while (!eof())
            {
                if (current == '*' && peekChar() == '/')
                {
                    nextChar(); nextChar();
                    break;
                }
                nextChar();
            }
        }
        else
        {
            break;
        }
        skipWhitespace(); // eat any space between consecutive comments
    }
}

Token Lexer::lexIdentifier()
{
    Position start = pos;
    std::string value;
    while(std::isalnum(static_cast<unsigned char>(current)) || current == '_')
    {
        value += current;
        nextChar();
    }
    auto it = KEYWORDS.find(value);
    TokenKind kind = (it != KEYWORDS.end()) ? it->second : TokenKind::IDENTIFIER;
    return makeToken(kind, value, start);
}

Token Lexer::lexNumber()
{
    Position start = pos;
    std::string value;
    bool isFloat = false;

    while(std::isdigit(static_cast<unsigned char>(current)))
    {
        value += current;
        nextChar();
    }

    if (current == '.' && std::isdigit(static_cast<unsigned char>(peekChar())))
    {
        isFloat = true;
        value += current; nextChar();
        while (std::isdigit(static_cast<unsigned char>(current)))
        {
            value += current;
            nextChar();
        }
    }

    if (current == 'e' || current == 'E')
    {
        isFloat = true;
        value += current; nextChar();
        if (current == '+' || current == '-')
        {
            value += current; nextChar();
        }
        while (std::isdigit(static_cast<unsigned char>(current)))
        {
            value += current;
            nextChar();
        }
    }

    if (current == 'i' || current == 'f' || current == 'u') // type suffix, e.g., 42i32, 3.14f64
    {
        char suffix_start = current;
        std::string suffix;
        suffix += current; nextChar();
        while (std::isdigit(static_cast<unsigned char>(current)))
        {
            suffix += current;
            nextChar();
        }
        if (suffix_start == 'f') isFloat = true;
        value += suffix;
    }

    return makeToken(isFloat ? TokenKind::FLOAT : TokenKind::INT, value, start);
}

Token Lexer::lexString()
{
    Position start = pos;
    std::string value;
    nextChar(); // skip opening "

    while(current != '"' && !eof())
    {
        if (current == '\\')
        {
            nextChar();
            switch (current)
            {
                case 'n':  value += '\n'; break;
                case 't':  value += '\t'; break;
                case 'r':  value += '\r'; break;
                case '\\': value += '\\'; break;
                case '"':  value += '"';  break;
                case '0':  value += '\0'; break;
                default:
                    value += '\\';
                    value += current;
                    break;
            }
        }
        else
        {
            value += current;
        }
        nextChar();
    }

    nextChar(); // skip closing "
    return makeToken(TokenKind::STRING, value, start);
}

Token Lexer::lexOperator()
{
    Position start = pos;
    char c = current;
    nextChar();

    switch(c)
    {
        // arithmetic
        case '+':
            if (current == '=') { nextChar(); return makeToken(TokenKind::PLUS_ASSIGN,  "+=", start); }
            return makeToken(TokenKind::PLUS,   "+", start);
        case '-':
            if (current == '>') { nextChar(); return makeToken(TokenKind::ARROW,         "->", start); }
            if (current == '=') { nextChar(); return makeToken(TokenKind::MINUS_ASSIGN,  "-=", start); }
            return makeToken(TokenKind::MINUS,  "-", start);
        case '*':
            if (current == '=') { nextChar(); return makeToken(TokenKind::STAR_ASSIGN,   "*=", start); }
            return makeToken(TokenKind::STAR,   "*", start);
        case '/':
            if (current == '=') { nextChar(); return makeToken(TokenKind::SLASH_ASSIGN,  "/=", start); }
            return makeToken(TokenKind::SLASH,  "/", start);
        case '@':
            return makeToken(TokenKind::MATMUL, "@", start);
        // comparison
        case '=':
            if (current == '=') { nextChar(); return makeToken(TokenKind::EQ,            "==", start); }
            if (current == '>') { nextChar(); return makeToken(TokenKind::FAT_ARROW,      "=>", start); }
            return makeToken(TokenKind::ASSIGN, "=", start);
        case '!':
            if (current == '=') { nextChar(); return makeToken(TokenKind::NEQ,            "!=", start); }
            return makeToken(TokenKind::NOT,    "!", start);
        case '<':
            if (current == '=') { nextChar(); return makeToken(TokenKind::LTE,            "<=", start); }
            if (current == '-') { nextChar(); return makeToken(TokenKind::CHANNEL_SEND,   "<-", start); }
            return makeToken(TokenKind::LT,     "<", start);
        case '>':
            if (current == '=') { nextChar(); return makeToken(TokenKind::GTE,            ">=", start); }
            return makeToken(TokenKind::GT,     ">", start);
        // logical
        case '&':
            if (current == '&') { nextChar(); return makeToken(TokenKind::AND,            "&&", start); }
            return makeToken(TokenKind::UNKNOWN, "&", start);
        case '|':
            if (current == '|') { nextChar(); return makeToken(TokenKind::OR,             "||", start); }
            if (current == '>') { nextChar(); return makeToken(TokenKind::PIPE,           "|>", start); }
            return makeToken(TokenKind::UNKNOWN, "|", start);
        // scope / type
        case ':':
            if (current == ':') { nextChar(); return makeToken(TokenKind::DOUBLE_COLON,   "::", start); }
            return makeToken(TokenKind::COLON,  ":", start);
        case '.': 
            // '...' must be checked before '..'
            if (current == '.' && peekChar() == '.')
            {
                nextChar(); nextChar();
                return makeToken(TokenKind::ELLIPSIS, "...", start);
            }
            if (current == '.')
            {
                nextChar();
                return makeToken(TokenKind::RANGE, "..", start);
            }
            return makeToken(TokenKind::DOT, ".", start);
        // generic sigilh
        case '#':
            return makeToken(TokenKind::HASH, "#", start);
        // brackets
        case '(':  return makeToken(TokenKind::LPAREN,    "(", start);
        case ')':  return makeToken(TokenKind::RPAREN,    ")", start);
        case '{':  return makeToken(TokenKind::LBRACE,    "{", start);
        case '}':  return makeToken(TokenKind::RBRACE,    "}", start);
        // '[' and ']' are handled in nextToken() for context tracking
        // They should never reach here, but guard just in case:
        case '[':  return makeToken(TokenKind::LBRACKET,  "[", start);
        case ']':  return makeToken(TokenKind::RBRACKET,  "]", start);
        // punctuation
        case ',':  return makeToken(TokenKind::COMMA,     ",", start);
        case ';':  return makeToken(TokenKind::SEMICOLON, ";", start);
        default:
            return makeToken(TokenKind::UNKNOWN, std::string(1, c), start);
    }
}

LexContext Lexer::currentContext() const
{
    return context_stack.empty() ? LexContext::DEFAULT : context_stack.back();
}

void Lexer::pushContext(LexContext ctx)
{
    context_stack.push_back(ctx);
}

void Lexer::popContext()
{
    if (!context_stack.empty())
        context_stack.pop_back();
}


Token Lexer::makeToken(TokenKind kind, const std::string& value, Position start)
{
    last_token_kind = kind;
    return Token(kind, value, start);
}

Token Lexer::nextToken()
{
    skipWhitespace();
    skipComment();
    if(eof())
        return Token(TokenKind::EOF_TOKEN,"",pos);
    // context-sensitive '{'. Map{...} → MAP context; anything else → DEFAULT (plain block)
    if (current == '{')
    {
        Position   start = pos;
        nextChar();
        LexContext ctx  = isBraceCollectionOpener(last_token_kind)
                        ? LexContext::MAP
                        : LexContext::DEFAULT;
        pushContext(ctx);
        return makeToken(TokenKind::LBRACE, "{", start);
    }
    if (current == '}')
    {
        Position start = pos;
        nextChar();
        popContext();
        return makeToken(TokenKind::RBRACE, "}", start);
    }
    // context-sensitive '['
    if (current == '[')
    {
        Position start = pos;
        nextChar();
        // '[' directly after Tensor keyword → tensor literal context
        LexContext ctx = (last_token_kind == TokenKind::KW_TENSOR ||
                          last_token_kind == TokenKind::RPAREN)   // Tensor#(...)[
                        ? LexContext::TENSOR
                        : LexContext::ARRAY;
        pushContext(ctx);
        return makeToken(TokenKind::LBRACKET, "[", start);
    }
    if (current == ']')
    {
        Position start = pos;
        nextChar();
        popContext();
        return makeToken(TokenKind::RBRACKET, "]", start);
    }
    // context-sensitive ';' inside Tensor[...] → ROW_SEP; everywhere else    → SEMICOLON
    if (current == ';')
    {
        Position start = pos;
        nextChar();
        TokenKind kind = (currentContext() == LexContext::TENSOR)
                       ? TokenKind::ROW_SEP
                       : TokenKind::SEMICOLON;
        return makeToken(kind, ";", start);
    }
    // standard dispatch
    if (std::isalpha(static_cast<unsigned char>(current)) || current == '_')
        return lexIdentifier();
    if (std::isdigit(static_cast<unsigned char>(current)))
        return lexNumber();
    if (current == '"')
        return lexString();
    return lexOperator();
}

Token Lexer::peekToken()
{
    const size_t        savedIndex        = index;
    const char          savedCurrent      = current;
    const Position      savedPos          = pos;
    const auto          savedContextStack = context_stack;
    const TokenKind     savedLastKind     = last_token_kind;

    Token tok = nextToken();

    index         = savedIndex;
    current       = savedCurrent;
    pos           = savedPos;
    context_stack = savedContextStack;
    last_token_kind = savedLastKind;

    return tok;
}