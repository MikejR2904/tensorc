#pragma once
 
#include "file.h"
 
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
 
// ANSI colour helpers (no external dep required).
// Disable by defining TENSORCERR_NO_COLOR before including this header.
#ifndef TENSORCERR_NO_COLOR
#  define _TC_RED     "\033[1;31m"
#  define _TC_BLUE    "\033[1;34m"
#  define _TC_YELLOW  "\033[1;33m"
#  define _TC_BOLD    "\033[1m"
#  define _TC_RESET   "\033[0m"
#else
#  define _TC_RED     ""
#  define _TC_BLUE    ""
#  define _TC_YELLOW  ""
#  define _TC_BOLD    ""
#  define _TC_RESET   ""
#endif
 
namespace io {
 
// ─── DiagnosticKind ──────────────────────────────────────────────────────────
 
enum class DiagnosticKind { Error, Help, Warn };
 
inline const char* kind_label(DiagnosticKind k)
{
    switch (k) {
        case DiagnosticKind::Error: return "error";
        case DiagnosticKind::Help:  return "help";
        case DiagnosticKind::Warn:  return "warning";
    }
    return "";
}
 
inline const char* kind_color(DiagnosticKind k)
{
    switch (k) {
        case DiagnosticKind::Error: return _TC_RED;
        case DiagnosticKind::Help:  return _TC_BLUE;
        case DiagnosticKind::Warn:  return _TC_YELLOW;
    }
    return _TC_RESET;
}
 
// ─── Diagnostic ──────────────────────────────────────────────────────────────
 
struct SpanInfo
{
    SourceLoc   loc;
    std::string source_line;   ///< the relevant line of source text
};
 
struct Diagnostic
{
    std::string                 msg;
    DiagnosticKind              kind;
    std::optional<std::vector<SpanInfo>> spans;
 
    // ── Constructors ─────────────────────────────────────────────────────────
 
    /// Diagnostic with no associated source span (e.g. CLI / I/O errors).
    static Diagnostic plain(std::string msg, DiagnosticKind kind)
    {
        return { std::move(msg), kind, std::nullopt };
    }
 
    /// Diagnostic with source spans. Resolves each `SourceLoc` to a
    /// `SpanInfo` by reading the relevant line from `src`.
    static Diagnostic with_spans(std::string          msg,
                                  DiagnosticKind       kind,
                                  const std::string&   src,
                                  std::vector<SourceLoc> locs)
    {
        std::vector<SpanInfo> infos;
        infos.reserve(locs.size());
 
        std::vector<std::string> lines;
        {
            std::istringstream ss(src);
            std::string ln;
            while (std::getline(ss, ln)) lines.push_back(ln);
        }
 
        for (auto& loc : locs) {
            size_t line_idx = (loc.span.beg.line > 0) ? loc.span.beg.line - 1 : 0;
            std::string line_text = (line_idx < lines.size()) ? lines[line_idx] : "";
            infos.push_back({ std::move(loc), std::move(line_text) });
        }
 
        return { std::move(msg), kind, std::move(infos) };
    }
 
    // ── Formatting ───────────────────────────────────────────────────────────
 
    std::string to_string() const
    {
        std::ostringstream out;
 
        // Header: "error: message"
        out << kind_color(kind)
            << kind_label(kind)
            << _TC_RESET
            << _TC_BOLD << ": " << _TC_RESET
            << msg;
 
        if (!spans.has_value()) return out.str();
 
        for (auto& si : *spans)
        {
            const auto& span    = si.loc.span;
            const auto& fp      = si.loc.file_path;
            std::string line_nr = std::to_string(span.beg.line);
            std::string indent(line_nr.size() + 1, ' ');
            std::string underline(span.len() > 0 ? span.len() : 1, '^');
 
            // "   at [line:col] in path/to/file.tc"
            out << "\n" << indent
                << _TC_BOLD << "at" << _TC_RESET
                << " [" << span.beg.line << ":" << span.beg.col << "] "
                << _TC_BOLD << "in" << _TC_RESET << " "
                << (fp ? fp->raw() : "<unknown>");
 
            // Source snippet
            out << "\n" << indent << "|\n"
                << _TC_BOLD << line_nr << _TC_RESET
                << " | " << si.source_line << "\n"
                << indent << "| "
                << std::string(span.beg.col > 1 ? span.beg.col - 1 : 0, ' ')
                << kind_color(kind) << _TC_BOLD << underline << _TC_RESET;
        }
 
        return out.str();
    }
 
    bool operator==(const Diagnostic& o) const
    {
        return msg == o.msg && kind == o.kind;
    }
};
 
// ─── ErrorFormatter ──────────────────────────────────────────────────────────
 
struct ErrorFormatter
{
    static std::string format(const std::vector<Diagnostic>& diags)
    {
        std::string out;
        for (size_t i = 0; i < diags.size(); ++i) {
            out += diags[i].to_string();
            if (i + 1 < diags.size()) out += "\n";
        }
        return out;
    }
};
 
// ─── TensorCError ────────────────────────────────────────────────────────────
 
/// The primary error object propagated through compilation stages.
/// Holds one or more `Diagnostic` objects.
struct TensorCError : std::exception
{
    std::vector<Diagnostic> diags;

    TensorCError() = default;
 
    explicit TensorCError(Diagnostic d)       : diags{std::move(d)} {}
    explicit TensorCError(std::vector<Diagnostic> ds) : diags(std::move(ds)) {}
 
    const char* what() const noexcept override
    {
        // Cached on first call.
        if (what_.empty()) what_ = ErrorFormatter::format(diags);
        return what_.c_str();
    }
 
    // ── Factory helpers ───────────────────────────────────────────────────
 
    static TensorCError syntax(const std::string& src,
                                const std::string& msg,
                                SourceLoc           loc)
    {
        return TensorCError(
            Diagnostic::with_spans(msg, DiagnosticKind::Error, src, {std::move(loc)})
        );
    }
 
    static TensorCError file_io(const std::string& msg)
    {
        return TensorCError(Diagnostic::plain(msg, DiagnosticKind::Error));
    }
 
    static TensorCError type_mismatch(const std::string& src,
                                       const std::string& expected,
                                       const std::string& got,
                                       SourceLoc           loc)
    {
        return TensorCError(Diagnostic::with_spans(
            "expected type '" + expected + "' but got type '" + got + "'",
            DiagnosticKind::Error, src, {std::move(loc)}
        ));
    }
 
    static TensorCError unknown_symbol(const std::string& src,
                                        const std::string& name,
                                        SourceLoc           loc)
    {
        return TensorCError(Diagnostic::with_spans(
            "unknown symbol '" + name + "'",
            DiagnosticKind::Error, src, {std::move(loc)}
        ));
    }
 
    static TensorCError args_mismatch(const std::string& src,
                                       size_t             expected_n,
                                       size_t             got_n,
                                       SourceLoc           loc)
    {
        const char* arg_str = expected_n == 1 ? "argument" : "arguments";
        return TensorCError(Diagnostic::with_spans(
            "expected " + std::to_string(expected_n) + " " + arg_str +
            " but received " + std::to_string(got_n),
            DiagnosticKind::Error, src, {std::move(loc)}
        ));
    }
 
    /// Tensor-specific: shape mismatch between operands.
    static TensorCError shape_mismatch(const std::string& src,
                                        const std::string& op,
                                        const std::string& lhs_shape,
                                        const std::string& rhs_shape,
                                        SourceLoc           loc)
    {
        return TensorCError(Diagnostic::with_spans(
            "shape mismatch in '" + op + "': " + lhs_shape + " vs " + rhs_shape,
            DiagnosticKind::Error, src, {std::move(loc)}
        ));
    }
 
    /// Tensor-specific: operation not differentiable where grad is required.
    static TensorCError not_differentiable(const std::string& src,
                                            const std::string& op,
                                            SourceLoc           loc)
    {
        return TensorCError(Diagnostic::with_spans(
            "'" + op + "' is not differentiable; cannot use in backward pass",
            DiagnosticKind::Error, src, {std::move(loc)}
        ));
    }
 
    /// Tensor-specific: parallel dispatch on a non-parallelisable op.
    static TensorCError not_parallelisable(const std::string& src,
                                            const std::string& op,
                                            SourceLoc           loc)
    {
        return TensorCError(Diagnostic::with_spans(
            "'" + op + "' cannot be dispatched in parallel",
            DiagnosticKind::Error, src, {std::move(loc)}
        ));
    }
 
private:
    mutable std::string what_;
};
 
// ─── Result<T> ───────────────────────────────────────────────────────────────
 
/// Minimal C++20-compatible Result type — no external dependencies.
///
/// Mirrors Rust's `Result<T, E>` pattern.
/// Usage:
///   io::Result<TypePtr> check_expr(...);
///   auto r = io::Ok(value);
///   auto r = io::Err(TensorCError::syntax(...));
///   if (r)          { use(*r);          }
///   else            { handle(r.error()); }
template<typename T>
class Result
{
public:
    // ── Constructors ─────────────────────────────────────────────────────────
    Result(T val)              : ok_(true),  val_(std::move(val)),           err_() {}
    Result(TensorCError err)   : ok_(false), val_(),                         err_(std::move(err)) {}
 
    // ── Observers ────────────────────────────────────────────────────────────
    explicit operator bool()  const noexcept { return ok_; }
    bool     has_value()      const noexcept { return ok_; }
 
    T&       operator*()       &  { return val_; }
    const T& operator*()  const&  { return val_; }
    T*       operator->()         { return &val_; }
    const T* operator->()   const { return &val_; }
 
    TensorCError&       error()       { return err_; }
    const TensorCError& error() const { return err_; }
 
    /// Unwrap or throw.
    T& value() {
        if (!ok_) throw err_;
        return val_;
    }
 
private:
    bool         ok_;
    T            val_;
    TensorCError err_;
};
 
/// Void specialisation — success carries no payload.
template<>
class Result<void>
{
public:
    Result()                   : ok_(true),  err_() {}
    Result(TensorCError err)   : ok_(false), err_(std::move(err)) {}
 
    explicit operator bool()  const noexcept { return ok_; }
    bool     has_value()      const noexcept { return ok_; }
 
    TensorCError&       error()       { return err_; }
    const TensorCError& error() const { return err_; }
 
    void value() const { if (!ok_) throw err_; }
 
private:
    bool         ok_;
    TensorCError err_;
};
 
/// Factory helpers — mirror Rust's Ok(...) / Err(...).
template<typename T> Result<T>    Ok(T val)          { return Result<T>(std::move(val)); }
inline              Result<void>  Ok()               { return Result<void>(); }
template<typename T> Result<T>    Err(TensorCError e) { return Result<T>(std::move(e)); }
inline              Result<void>  Err(TensorCError e) { return Result<void>(std::move(e)); }
 
} // namespace io
 
#undef _TC_RED
#undef _TC_BLUE
#undef _TC_YELLOW
#undef _TC_BOLD
#undef _TC_RESET
 