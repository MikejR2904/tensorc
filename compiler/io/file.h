#pragma once

#include <string>
#include <memory>
#include <fstream>
#include <sstream>
#include <stdexcept>
 
// We are interested in the file's path and contents just as python's open() or builtin functions would provide.
namespace io {

struct FilePath
{
    explicit FilePath(std::string path) : path_(std::move(path)) {}
    const std::string& raw() const noexcept { return path_; }
    std::string str() const { return path_; }
 
    bool operator==(const FilePath& o) const noexcept { return path_ == o.path_; }
    bool operator!=(const FilePath& o) const noexcept { return path_ != o.path_; }
 
private:
    std::string path_;
};
 
/// A (line, col) position inside a source file. Both are 1-based.
struct FilePos
{
    size_t line = 1;
    size_t col = 1;
    static FilePos default_() noexcept { return {1, 1}; }
 
    bool operator==(const FilePos& o) const noexcept { return line == o.line && col == o.col; }
};
 
/// A half-open [beg, end) span of source text.
struct FileSpan
{
    FilePos beg;
    FilePos end;
    /// Length in columns (single-line spans only).
    size_t len() const noexcept { return end.col > beg.col ? end.col - beg.col : 0; }
    /// Construct a span of length 1
    static FileSpan one(FilePos pos) noexcept { return { pos, { pos.line, pos.col + 1 } }; }
    /// Extend this span's end to match other's end.
    FileSpan to(const FileSpan& other) const noexcept { return { beg, other.end }; }
    /// A dummy span for nodes without meaningful location information.
    static FileSpan dummy() noexcept { return { {1, 1}, {1, 1} }; }
 
    bool operator==(const FileSpan& o) const noexcept { return beg == o.beg && end == o.end; }
};
 
/// Source location associates a file span with the file it lives in.
struct SourceLoc
{
    std::shared_ptr<FilePath> file_path;
    FileSpan span;
    SourceLoc() = default;
    SourceLoc(std::shared_ptr<FilePath> fp, FileSpan sp) : file_path(std::move(fp)), span(sp) {}
 
    bool operator==(const SourceLoc& o) const noexcept
    {
        return span == o.span && (file_path == o.file_path || (file_path && o.file_path && *file_path == *o.file_path));
    }
};
 
/// Owns the source text of a single file and provides repeated access.
class FileHandler
{
public:
    /// Open `path` and read its full UTF-8 contents. Throws `std::runtime_error` on failure.
    explicit FileHandler(FilePath path) : file_path_(std::make_shared<FilePath>(std::move(path)))
    {
        std::ifstream f(file_path_->raw(), std::ios::binary);
        if (!f.is_open()) throw std::runtime_error("cannot open '" + file_path_->raw() + "'");
        std::ostringstream ss;
        ss << f.rdbuf();
        src_ = ss.str();
    }
    explicit FileHandler(const std::string& path) : FileHandler(FilePath(path)) {}
 
    const std::string& contents() const noexcept { return src_; }
    const std::shared_ptr<FilePath>& file_path() const noexcept { return file_path_; }
    bool operator==(const FileHandler& o) const noexcept { return *file_path_ == *o.file_path_; }
 
private:
    std::shared_ptr<FilePath> file_path_;
    std::string src_;
};
 
}