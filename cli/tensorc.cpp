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
#include "../compiler/io/io.h"
#include "../compiler/ir/ir.h"

// ── helpers ──────────────────────────────────────────────────────────────────
 
static void print_usage()
{
    std::cout
        << "Usage: tensorc <file.tcc> [options]\n"
        << "Options:\n"
        << "  -h, --help      Show this help message\n"
        << "  -v, --version   Display compiler version\n"
        << "\n"
        << "Example:\n"
        << "  tensorc hello_world.tcc --print-ir\n";
}
 
// ── entry point ──────────────────────────────────────────────────────────────
 
int main(int argc, char** argv)
{
    if (argc < 2) {
        print_usage();
        return 1;
    }
 
    std::string first_arg = argv[1];
    bool print_ir = false;

    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--print-ir") print_ir = true;
    }
 
    if (first_arg == "-h" || first_arg == "--help") {
        print_usage();
        return 0;
    }
 
    if (first_arg == "-v" || first_arg == "--version") {
        std::cout << "TensorC Compiler v0.1.0-alpha\n";
        return 0;
    }
 
    const std::string& filepath = first_arg;
    if (filepath.size() < 4 || filepath.substr(filepath.size() - 4) != ".tcc") {
        std::cerr << "tensorc: input file must be .tcc\n";
        return 1;
    }
 
    try {
        auto t0 = std::chrono::high_resolution_clock::now();
 
        // ── 1. I/O phase ─────────────────────────────────────────────────────
        // FileHandler owns the source text and its path.
        // Throws io::TensorCError (caught below) on missing/unreadable files.
        io::FileHandler fh(filepath);
        const std::string& source = fh.contents();
 
        // ── 2. Lexical analysis ───────────────────────────────────────────────
        Lexer lexer(source);
 
        // ── 3. Syntax analysis ────────────────────────────────────────────────
        Parser parser(lexer);
        auto program = parser.parse();
 
        if (program.stmts.empty()) {
            std::cerr << "tensorc: parser returned empty AST\n";
            return 1;
        }
 
        // ── 4. Semantic analysis ──────────────────────────────────────────────
        // BuiltinRegistry is constructed once here and moved into the analyser,
        // keeping ownership explicit at the top of the compilation pipeline.
        io::BuiltinRegistry builtins = io::BuiltinRegistry::with_builtins();
        SemanticAnalyzer sema(std::move(builtins));
        sema.validate(program);

        ir::IRBuilder builder;
        auto module = std::make_unique<ir::IRModule>(filepath);
        builder.build(program, module.get());

 
        auto t1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> ms = t1 - t0;
 
        std::cout << "[TensorC] Compilation successful ("
                  << ms.count() << "ms)\n";

        if (print_ir) {
            std::cout << "\n--- Generated IR ---\n";
            std::cout << ir::IRPrinter::print(*module) << "\n";
        }
 
    } catch (const io::TensorCError& e) {
        // Structured compiler diagnostics — already formatted with source spans.
        std::cerr << e.what() << "\n";
        return 1;
    } catch (const std::exception& e) {
        // Fallback for anything not yet ported to TensorCError.
        std::cerr << "tensorc: fatal error: " << e.what() << "\n";
        return 1;
    }
 
    return 0;
}
 