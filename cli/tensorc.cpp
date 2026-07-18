// Build this using: g++ -std=c++17 -I./compiler/ -o cli/tensorc cli/tensorc.cpp compiler/lexer/Lexer.cpp compiler/parser/Parser.cpp compiler/ast/Type.cpp
// Example usage: cli\tensorc.exe .\cli\test_files\test_1.tcc --print-ir

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
#include "../compiler/io/module_handler.h"
#include "../compiler/ir/ir.h"
#include "../codegen/CodegenDriver.h"

static void print_usage()
{
    std::cout
        << "Usage: tensorc <file.tcc> [options]\n"
        << "Options:\n"
        << "  -h, --help      Show this help message\n"
        << "  -v, --version   Display compiler version\n"
        << "  --print-ir      Print generated IR\n"
        << "  --emit=asm      Emit target assembly\n"
        << "  --target=<arch> Select codegen target (riscv64 or x86_64)\n"
        << "  -o <path>       Output path for emitted assembly\n"
        << "\n"
        << "Example:\n"
        << "  tensorc hello_world.tcc --print-ir\n"
        << "  tensorc hello_world.tcc --emit=asm --target=x86_64 -o hello.s\n";
}
 
int main(int argc, char** argv)
{
    if (argc < 2) {
        print_usage();
        return 1;
    }
    std::string first_arg = argv[1];
    bool print_ir = false;
    bool emit_asm = false;
    std::string target = "riscv64";
    std::string output_path;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--print-ir") {
            print_ir = true;
        } else if (arg == "--emit=asm" || arg == "--emit-asm") {
            emit_asm = true;
        } else if (arg.rfind("--target=", 0) == 0) {
            target = arg.substr(std::string("--target=").size());
        } else if (arg == "-o" && i + 1 < argc) {
            output_path = argv[++i];
        }
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
 
        // I/O phase: read source file into memory
        // FileHandler owns the source text and its path.
        // Throws io::TensorCError (caught below) on missing/unreadable files.
        io::FileHandler fh(filepath);
        const std::string& source = fh.contents();
 
        // Lexical analysis
        Lexer lexer(source);
 
        // Syntax analysis
        Parser parser(lexer);
        auto program = parser.parse();
 
        if (program.stmts.empty()) {
            std::cerr << "tensorc: parser returned empty AST\n";
            return 1;
        }
 
        // Semantic analysis
        // BuiltinRegistry is constructed once here and moved into the analyser, keeping ownership explicit at the top of the compilation pipeline.
        io::BuiltinRegistry builtins = io::BuiltinRegistry::with_builtins();
        io::ModuleHandlerRegistry handlers = io::ModuleHandlerRegistry::with_builtins();
        SemanticAnalyzer sema(builtins);
        sema.validate(program);

        // IR generation
        ir::IRBuilder builder;
        auto module = std::make_unique<ir::IRModule>(filepath);
        builder.build(program, module.get(), builtins, handlers);
        int fusions = ir::PassPipeline::run(*module);
 
        auto t1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> ms = t1 - t0;
 
        std::cout << "[TensorC] Compilation successful ("
                  << ms.count() << "ms)\n";

        if (print_ir) {
            std::cout << "\n--- Generated IR ---\n";
            std::cout << ir::IRPrinter::print(*module) << "\n";
        }

        if (emit_asm) {
            if (output_path.empty()) {
                output_path = filepath.substr(0, filepath.size() - 4) + ".s";
            }
            codegen::CodegenDriver driver(target);
            if (!driver.lower_module_to_asm(*module, output_path)) {
                std::cerr << "tensorc: codegen failed: " << driver.last_diagnostic() << "\n";
                return 1;
            }
            std::cout << "[TensorC] Wrote assembly: " << output_path << "\n";
        }
 
    } catch (const io::TensorCError& e) {
        // Structured compiler diagnostics, already formatted with source spans.
        std::cerr << e.what() << "\n";
        return 1;
    } catch (const std::exception& e) {
        // Fallback for anything not yet ported to TensorCError.
        std::cerr << "tensorc: fatal error: " << e.what() << "\n";
        return 1;
    }
 
    return 0;
}
 
