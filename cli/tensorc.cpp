#include <fstream>
#include <string>
#include <iostream>
#include <vector>
#include <memory>
#include <chrono>

#include "../compiler/lexer/Lexer.h"
#include "../compiler/parser/Parser.h"
#include "../compiler/ast/ASTNode.h"
#include "../compiler/ast/SemanticAnalyzer.h"

static std::string read_file(const std::string& path) {
    // Open at the end to get the size immediately
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    
    if (!file.is_open()) {
        std::cerr << "tensorc: cannot open file: " << path << std::endl;
        std::exit(1);
    }

    std::streamsize size = file.tellg();
    if (size <= 0) return ""; // Handle empty files gracefully

    file.seekg(0, std::ios::beg);

    std::string buffer;
    buffer.resize(static_cast<size_t>(size));

    if (!file.read(&buffer[0], size)) {
        std::cerr << "tensorc: error reading file content: " << path << std::endl;
        std::exit(1);
    }

    return buffer;
}

static void print_usage() {
    std::cout << "Usage: tensorc <file.ctx> [options]\n"
              << "Options:\n"
              << "  -h, --help      Show this help message\n"
              << "  -v, --version   Display compiler version\n"
              << "\n"
              << "Example:\n"
              << "  tensorc hello_world.tcc\n";
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage();
        return 1;
    }

    std::string first_arg = argv[1];
    if (first_arg == "-h" || first_arg == "--help") {
        print_usage();
        return 0;
    }

    if (first_arg == "-v" || first_arg == "--version") {
        std::cout << "TensorC Compiler v0.1.0-alpha\n";
        return 0;
    }

    std::string filepath = first_arg;
    if (filepath.size() < 4 || filepath.substr(filepath.size() - 4) != ".tcc") {
        std::cerr << "tensorc: input file must be .tcc\n";
        return 1;
    }

    try {
        auto start = std::chrono::high_resolution_clock::now();

        // 1. I/O Phase
        std::string source = read_file(filepath);
        
        // 2. Lexical Analysis
        Lexer lexer(source);
        
        // 3. Syntax Analysis
        Parser parser(lexer);
        auto program = parser.parse();

        if (program.stmts.empty()) {
            std::cerr << "tensorc: parser returned null AST\n";
            return 1;
        }

        // 4. Semantic Analysis
        SemanticAnalyzer sema; 
        sema.validate(program);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;

        std::cout << "[TensorC] Compilation successful (" << duration.count() << "ms)\n";

    } catch (const std::exception& e) {
        std::cerr << "tensorc: fatal error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
