#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include "Lexer.h"

// Helper function to turn TokenKind enum into a readable string
std::string kindToString(TokenKind k) {
    switch(k) {
        // Essentials & Literals
        case TokenKind::EOF_TOKEN:    return "EOF";
        case TokenKind::IDENTIFIER:   return "IDENTIFIER";
        case TokenKind::NUMBER:       return "NUMBER";
        case TokenKind::STRING:       return "STRING";
        case TokenKind::FLOAT:        return "FLOAT";
        case TokenKind::INT:          return "INT";

        // Control Flow Keywords
        case TokenKind::KW_IF:        return "KW_IF";
        case TokenKind::KW_ELSE:      return "KW_ELSE";
        case TokenKind::KW_FOR:       return "KW_FOR";
        case TokenKind::KW_WHILE:     return "KW_WHILE";
        case TokenKind::KW_BREAK:     return "KW_BREAK";
        case TokenKind::KW_CONTINUE:  return "KW_CONTINUE";
        case TokenKind::KW_MATCH:     return "KW_MATCH";
        case TokenKind::KW_RETURN:    return "KW_RETURN";

        // Declarations
        case TokenKind::KW_FN:        return "KW_FN";
        case TokenKind::KW_LET:       return "KW_LET";
        case TokenKind::KW_PUB:       return "KW_PUB";

        // Types & ML
        case TokenKind::KW_I32:       return "KW_I32";
        case TokenKind::KW_I64:       return "KW_I64";
        case TokenKind::KW_F32:       return "KW_F32";
        case TokenKind::KW_F64:       return "KW_F64";
        case TokenKind::KW_BOOL:      return "KW_BOOL";
        case TokenKind::KW_TRUE:      return "KW_TRUE";
        case TokenKind::KW_FALSE:     return "KW_FALSE";
        case TokenKind::KW_STR:       return "KW_STR";
        case TokenKind::KW_VOID:      return "KW_VOID";
        case TokenKind::KW_TENSOR:    return "KW_TENSOR";
        case TokenKind::KW_GRAD:      return "KW_GRAD";
        case TokenKind::KW_STACK:     return "KW_STACK";
        case TokenKind::KW_QUEUE:     return "KW_QUEUE";
        case TokenKind::KW_TUPLE:     return "KW_TUPLE";
        case TokenKind::KW_MAP:       return "KW_MAP";
        case TokenKind::KW_SET:       return "KW_SET";

        // Async & Distributed
        case TokenKind::KW_ASYNC:     return "KW_ASYNC";
        case TokenKind::KW_AWAIT:     return "KW_AWAIT";
        case TokenKind::KW_SPAWN:     return "KW_SPAWN";
        case TokenKind::KW_YIELD:     return "KW_YIELD";

        // Modules
        case TokenKind::KW_IMPORT:    return "KW_IMPORT";
        case TokenKind::KW_FROM:      return "KW_FROM";
        case TokenKind::KW_AS:        return "KW_AS";
        case TokenKind::KW_IN:        return "KW_IN";
        case TokenKind::KW_STRUCT:    return "KW_STRUCT";

        // Arithmetic
        case TokenKind::PLUS:         return "PLUS";
        case TokenKind::MINUS:        return "MINUS";
        case TokenKind::STAR:         return "STAR";
        case TokenKind::SLASH:        return "SLASH";
        case TokenKind::MATMUL:       return "MATMUL (@)";

        // Augmented Assignment
        case TokenKind::PLUS_ASSIGN:  return "PLUS_ASSIGN";
        case TokenKind::MINUS_ASSIGN: return "MINUS_ASSIGN";
        case TokenKind::STAR_ASSIGN:  return "STAR_ASSIGN";
        case TokenKind::SLASH_ASSIGN: return "SLASH_ASSIGN";

        // Comparison & Logic
        case TokenKind::ASSIGN:       return "ASSIGN (=)";
        case TokenKind::EQ:           return "EQ (==)";
        case TokenKind::NEQ:          return "NEQ (!=)";
        case TokenKind::LT:           return "LT";
        case TokenKind::GT:           return "GT";
        case TokenKind::LTE:          return "LTE";
        case TokenKind::GTE:          return "GTE";
        case TokenKind::AND:          return "AND (&&)";
        case TokenKind::OR:           return "OR (||)";
        case TokenKind::NOT:          return "NOT (!)";

        // Specialized Operators
        case TokenKind::ARROW:        return "ARROW (->)";
        case TokenKind::FAT_ARROW:    return "FAT_ARROW (=>)";
        case TokenKind::PIPE:         return "PIPE (|>)";
        case TokenKind::CHANNEL_SEND: return "CHANNEL_SEND (<-)";
        case TokenKind::HASH:         return "HASH (#)";

        // Structure & Scoping
        case TokenKind::DOUBLE_COLON: return "DOUBLE_COLON (::)";
        case TokenKind::DOT:          return "DOT (.)";
        case TokenKind::RANGE:        return "RANGE (..)";
        case TokenKind::ELLIPSIS:     return "ELLIPSIS (...)";

        // Delimiters
        case TokenKind::LPAREN:       return "LPAREN";
        case TokenKind::RPAREN:       return "RPAREN";
        case TokenKind::LBRACE:       return "LBRACE";
        case TokenKind::RBRACE:       return "RBRACE";
        case TokenKind::LBRACKET:     return "LBRACKET";
        case TokenKind::RBRACKET:     return "RBRACKET";

        // Punctuation & Contextual
        case TokenKind::COMMA:        return "COMMA";
        case TokenKind::COLON:        return "COLON";
        case TokenKind::SEMICOLON:    return "SEMICOLON";
        case TokenKind::ROW_SEP:      return "ROW_SEP (;)";

        case TokenKind::UNKNOWN:      return "UNKNOWN";
        default:                      return "UNHANDLED_TOKEN";
    }
}
int main() {
    std::string code = R"(
        let weights = Tensor#([3, 3], f32) [
            1.0, 0.0, 0.5;
            0.2, 1.1, 0.0;
            0.0, 0.5, 1.0
        ];
        fn main() { return 0; }
        let set1 = Set[0.1, 0.2, 0.3]
        let queue2 = Queue[0.3, 0.5, 0.2]
        let map3 = Map{"k1": 0.32, "k2": 100}

        struct Point { x: f32, y: f32 }
    )";

    Lexer lexer(code);

    std::cout << "Tokenizing TensorC Code..." << std::endl;
    std::cout << "---------------------------" << std::endl;

    std::cout << std::left << std::setw(10) << "POS" 
              << std::setw(15) << "KIND" 
              << "VALUE" << std::endl;
    std::cout << std::string(40, '-') << std::endl;

    while (true) {
        // Construct directly from nextToken() - no default constructor needed
        Token tok = lexer.nextToken();

        std::cout << "[" << tok.pos.line << ":" << tok.pos.column << "] " 
                  << std::left << std::setw(15) << kindToString(tok.kind) 
                  << "'" << tok.value << "'" << std::endl;

        if (tok.kind == TokenKind::EOF_TOKEN) break;
    }

    return 0;
}