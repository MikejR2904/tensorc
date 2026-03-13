#pragma once
 
#include "file.h"
#include "error.h"
#include "cache.h"
#include "builtins.h"
#include "../ast/SemanticAnalyzer.h"
#include "../lexer/Lexer.h"
#include "../parser/Parser.h"
 
#include <filesystem>
#include <string>
#include <unordered_set>
#include <vector>
 
// Forward-declare the compiler stages so ImportResolver.h doesn't need
// to pull in their full headers (avoids circular deps at this layer).
// The .cpp implementation includes them.
class Lexer;
class Parser;
class SemanticAnalyzer;
 
namespace io {
 
namespace fs = std::filesystem;
 
// ─── ImportRequest ────────────────────────────────────────────────────────────
 
/// One parsed import statement.
struct ImportRequest
{
    std::string raw_path;     ///< exactly as written in source, e.g. "./utils"
    std::string alias;        ///< the name visible in the importing module
    bool        is_builtin;   ///< true for std / math / tensor / nn / optim / …
    fs::path    resolved;     ///< absolute path (empty for builtins)
};
 
// ─── ImportResolver ──────────────────────────────────────────────────────────
 
class ImportResolver
{
public:
    /// `registry`   — the BuiltinRegistry for the current session.
    ///                Resolved user modules are injected into it under their alias.
    /// `origin_dir` — directory of the importing file; used to resolve relative paths.
    explicit ImportResolver(BuiltinRegistry& registry,
                            fs::path         origin_dir = fs::current_path())
        : registry_(registry)
        , origin_dir_(std::move(origin_dir))
    {}
 
    // ── Main entry point ──────────────────────────────────────────────────────
 
    /// Resolve a single import statement.
    /// Returns the canonical path (or module name for builtins), or an error.
    ///
    /// On success, the module's symbols are accessible via:
    ///   registry_.lookup(alias, symbol_name)
    io::Result<std::string> resolve(const ImportRequest& req)
    {
        // ── 1. Built-ins are pre-registered; nothing to do ───────────────────
        if (req.is_builtin)
        {
            if (!registry_.has_module(req.alias))
                return io::Err<std::string>(TensorCError::file_io(
                    "unknown built-in module '" + req.raw_path + "'"));
            return io::Ok(req.raw_path);
        }
 
        // ── 2. Canonicalise path ─────────────────────────────────────────────
        fs::path abs_path;
        try {
            abs_path = fs::canonical(origin_dir_ / req.resolved);
        } catch (...) {
            return io::Err<std::string>(TensorCError::file_io(
                "cannot resolve import path '" + req.raw_path + "'"));
        }
 
        const std::string key = abs_path.string();
 
        // ── 3. Cycle detection ───────────────────────────────────────────────
        if (in_progress_.count(key))
            return io::Err<std::string>(TensorCError::file_io(
                "circular import detected: '" + key + "'"));
 
        // ── 4. Cache check ───────────────────────────────────────────────────
        // Read the source first to compute its hash.
        io::FileHandler fh(key);
        const std::string& src = fh.contents();
        ContentHash hash = ContentHash::of(src);
 
        if (auto cached = GlobalModuleCache::instance().get(key, hash))
        {
            // Cache hit — just inject the exports into the registry.
            if (cached->exports)
                registry_.add(req.alias, cached->exports);
            return io::Ok(key);
        }
 
        // ── 5. Full pipeline ─────────────────────────────────────────────────
        in_progress_.insert(key);
 
        auto result = run_pipeline(key, src, req.alias, hash, abs_path.parent_path());
 
        in_progress_.erase(key);
        return result;
    }
 
    /// Resolve multiple imports (e.g. all imports at the top of a file).
    /// Stops at the first error.
    io::Result<void> resolve_all(const std::vector<ImportRequest>& requests)
    {
        for (auto& req : requests)
        {
            auto r = resolve(req);
            if (!r) return io::Err(std::move(r.error()));
        }
        return io::Ok();
    }
 
    /// Flush the thread-local cache into the global cache.
    /// Call this when the owning thread finishes all its import work.
    void flush()
    {
        GlobalModuleCache::instance().merge(local_cache_);
        local_cache_.clear();
    }
 
    // ── Path parsing helpers ──────────────────────────────────────────────────
 
    static ImportRequest parse_import(const std::string& stmt,
                                      const fs::path&    origin_dir)
    {
        static const std::unordered_set<std::string> BUILTINS =
            { "std", "math", "tensor", "nn", "optim", "data", "parallel" };
 
        ImportRequest req;
 
        // Split on " as "
        std::string path_part, alias_part;
        auto as_pos = stmt.find(" as ");
        if (as_pos != std::string::npos) {
            path_part  = trim(stmt.substr(0, as_pos));
            alias_part = trim(stmt.substr(as_pos + 4));
        } else {
            path_part  = trim(stmt);
        }
 
        // Strip surrounding quotes for file-path imports
        if (!path_part.empty() && path_part.front() == '"')
            path_part = path_part.substr(1, path_part.size() - 2);
 
        req.raw_path   = path_part;
        req.is_builtin = BUILTINS.count(path_part) > 0;
 
        if (req.is_builtin) {
            req.alias    = alias_part.empty() ? path_part : alias_part;
            req.resolved = "";
        } else {
            // Derive alias from file stem if not given
            fs::path p(path_part);
            req.alias    = alias_part.empty() ? p.stem().string() : alias_part;
            // Ensure .tcc extension
            if (!p.has_extension()) p.replace_extension(".tcc");
            req.resolved = p;
        }
 
        return req;
    }
 
private:
    BuiltinRegistry&             registry_;
    fs::path                     origin_dir_;
    ThreadLocalCache             local_cache_;
    std::unordered_set<std::string> in_progress_;  ///< cycle detection
 
    // ── Full pipeline ─────────────────────────────────────────────────────────
 
    io::Result<std::string> run_pipeline(const std::string& key,
                                          const std::string& src,
                                          const std::string& alias,
                                          const ContentHash& hash,
                                          const fs::path&    module_dir)
    {
        // ── 5a. Lex ──────────────────────────────────────────────────────────
        Lexer lexer(src);
 
        // ── 5b. Parse ────────────────────────────────────────────────────────
        Parser parser(lexer);
        auto program = parser.parse();
 
        if (program.stmts.empty())
            return io::Err<std::string>(TensorCError::file_io(
                "module '" + key + "' parsed to an empty AST"));
 
        // ── 5c. Resolve nested imports first (recursive) ─────────────────────
        ImportResolver nested(registry_, module_dir);
        auto nested_imports = collect_imports(program, module_dir);
        auto nested_result  = nested.resolve_all(nested_imports);
        if (!nested_result)
            return io::Err<std::string>(std::move(nested_result.error()));
        nested.flush();
 
        // ── 5d. Semantic analysis ─────────────────────────────────────────────
        // We construct a fresh SemanticAnalyzer that inherits the now-updated
        // registry (which includes the nested module symbols).
        SemanticAnalyzer sema(registry_);
        try {
            sema.validate(program);
        } catch (const std::exception& e) {
            return io::Err<std::string>(TensorCError::file_io(
                std::string("semantic error in '") + key + "': " + e.what()));
        }
 
        // ── 5e. Extract exports ───────────────────────────────────────────────
        auto exports = extract_exports(program, key);
 
        // ── 5f. Inject into registry ──────────────────────────────────────────
        registry_.add(alias, exports);
 
        // ── 5g. Cache ────────────────────────────────────────────────────────
        CachedModule entry;
        entry.path    = key;
        entry.hash    = hash;
        entry.source  = src;
        entry.exports = exports;
        // AST subtree stored for incremental re-use by later passes.
        // (ast_root would be set here once IRModule is available)
 
        local_cache_.put(key, std::move(entry));
 
        return io::Ok(key);
    }
 
    // ── Helpers ───────────────────────────────────────────────────────────────
 
    /// Walk the top-level AST and collect all import statements.
    /// Implemented in ImportResolver.cpp — declared here for the header.
    static std::vector<ImportRequest> collect_imports(
        const struct Program& program, const fs::path& origin_dir);
 
    /// Walk the sema-annotated AST and build a ModuleExports from all
    /// top-level `fn`, `let`, `enum`, and `struct` definitions marked exported.
    static std::shared_ptr<ModuleExports> extract_exports(
        const struct Program& program, const std::string& path);
 
    static std::string trim(const std::string& s)
    {
        size_t l = s.find_first_not_of(" \t\r\n");
        size_t r = s.find_last_not_of(" \t\r\n");
        return (l == std::string::npos) ? "" : s.substr(l, r - l + 1);
    }
};
 
} // namespace io