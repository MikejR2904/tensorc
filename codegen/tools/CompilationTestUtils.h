/// codegen/tools/CompilationTestUtils.h
///
/// Compiles real TensorC *source text* through the exact pipeline
/// cli/tensorc.cpp uses — Lexer -> Parser -> SemanticAnalyzer -> IRBuilder ->
/// PassPipeline -> ScalarCodegenPipeline — and returns assembly text.
///
/// This is the piece REAL_EXECUTION_TESTING_GUIDE.md originally sketched as
/// `compile_to_assembly()` against an invented API (`Lexer::tokenize`,
/// `IRBuilder::build(ast)` with no module, `CodegenDriver::generate_assembly`)
/// that never matched the real compiler. This version calls the actual
/// classes with their actual signatures — see cli/tensorc.cpp for the
/// reference sequence this mirrors.
///
/// Combine with codegen/tools/ExecUtils.h (assemble + link + load, verified
/// against the local x86_64-w64-mingw32 toolchain) to go all the way from
/// source text to a callable function pointer.

#pragma once

#include "../../compiler/lexer/Lexer.h"
#include "../../compiler/parser/Parser.h"
#include "../../compiler/ast/SemanticAnalyzer.h"
#include "../../compiler/io/io.h"
#include "../../compiler/io/module_handler.h"
#include "../../compiler/ir/ir.h"
#include "../ScalarCodegenPipeline.h"
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

namespace codegen::testing {

/// Compiles `source` (a complete .tcc program, not a snippet — it goes
/// through the real parser, so it needs top-level `fn`/`import`/etc. just
/// like a file would) and returns the assembly for every function with a
/// body, concatenated in module order. `target` is "x86_64" or "riscv64".
///
/// Throws std::runtime_error (semantic errors surface as io::TensorCError,
/// which derives from it) on any pipeline failure — tests should let that
/// propagate rather than silently getting an empty string back, so a broken
/// test points at the actual failing stage.
inline std::string compile_source_to_asm(const std::string& source, const std::string& target = "x86_64") {
    Lexer lexer(source);
    Parser parser(lexer);
    Program program = parser.parse();
    if (program.stmts.empty()) {
        throw std::runtime_error("compile_source_to_asm: parser returned an empty program");
    }

    io::BuiltinRegistry builtins = io::BuiltinRegistry::with_builtins();
    io::ModuleHandlerRegistry handlers = io::ModuleHandlerRegistry::with_builtins();
    SemanticAnalyzer sema(builtins);
    sema.validate(program);

    ir::IRBuilder builder;
    auto module = std::make_unique<ir::IRModule>("<compile_source_to_asm>");
    builder.build(program, module.get(), builtins, handlers);
    ir::PassPipeline::run(*module);

    codegen::ScalarCodegenPipeline pipeline(target);
    if (!pipeline.valid()) {
        throw std::runtime_error("compile_source_to_asm: unknown codegen target '" + target + "'");
    }

    std::ostringstream oss;
    for (auto& fn : module->functions) {
        if (fn->blocks.empty()) continue; // declaration-only builtin stubs (math::sin, tensor::*, ...)
        if (!pipeline.lower_function(*fn, oss)) {
            throw std::runtime_error("compile_source_to_asm: codegen failed for function '" + fn->name + "'");
        }
        oss << "\n";
    }
    return oss.str();
}

} // namespace codegen::testing
