#pragma once
 
#include "../ast/SymbolTable.h"
#include "../ast/Type.h"
#include "file.h"
#include "builtins.h"
 
#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>
 
namespace io {
 
// ─── Forward declarations ────────────────────────────────────────────────────
 
struct ASTNode;   // defined in ast/; cache holds shared_ptr<ASTNode>
 
// ─── Content hash ────────────────────────────────────────────────────────────
 
/// A 256-bit (32-byte) content hash computed over raw source bytes.
struct ContentHash
{
    std::array<uint8_t, 32> bytes{};
 
    bool operator==(const ContentHash& o) const noexcept { return bytes == o.bytes; }
    bool operator!=(const ContentHash& o) const noexcept { return bytes != o.bytes; }
 
    /// Compute a fast, non-cryptographic 256-bit hash of `src`.
    /// Uses four interleaved FNV-1a 64-bit streams to fill all 32 bytes,
    /// which is sufficient for cache-invalidation purposes.
    static ContentHash of(const std::string& src) noexcept
    {
        constexpr uint64_t FNV_PRIME  = 0x00000100000001B3ULL;
        constexpr uint64_t FNV_OFFSET = 0xCBF29CE484222325ULL;
 
        // Four independent accumulators with distinct seeds.
        uint64_t h[4] = {
            FNV_OFFSET,
            FNV_OFFSET ^ 0xDEADBEEFCAFEBABEULL,
            FNV_OFFSET ^ 0x0123456789ABCDEFULL,
            FNV_OFFSET ^ 0xFEDCBA9876543210ULL,
        };
 
        for (size_t i = 0; i < src.size(); ++i)
        {
            h[i & 3] ^= static_cast<uint8_t>(src[i]);
            h[i & 3] *= FNV_PRIME;
        }
 
        ContentHash out;
        for (int i = 0; i < 4; ++i)
            std::memcpy(out.bytes.data() + i * 8, &h[i], 8);
        return out;
    }
 
    /// Hex string representation (for diagnostics / debug output).
    std::string to_hex() const
    {
        static constexpr char HEX[] = "0123456789abcdef";
        std::string s;
        s.reserve(64);
        for (uint8_t b : bytes) {
            s += HEX[b >> 4];
            s += HEX[b & 0xF];
        }
        return s;
    }
};
 
// ─── Tensor op metadata ──────────────────────────────────────────────────────
 
/// Shape dimensions; -1 means "dynamic / unknown at compile time".
using Shape = std::vector<int64_t>;
 
/// Metadata for a single tensor operation exported by a module.
struct TensorOpMeta
{
    std::string         name;
    std::vector<Shape>  input_shapes;   ///< one Shape per positional argument
    Shape               output_shape;
    bool                differentiable  = false;  ///< supports backward()
    bool                parallelisable  = true;   ///< safe to dispatch across threads
    std::string         device_hint;              ///< e.g. "cpu", "cuda", "any"
};
 
// ─── Cached module entry ─────────────────────────────────────────────────────
 
/// Everything the compiler knows about one source module after it has been
/// fully processed. All fields are optional so the cache can be partially
/// populated (e.g. after lexing but before type-checking).
struct CachedModule
{
    // ── Identity ─────────────────────────────────────────────────────────────
    std::string  path;          ///< canonical absolute path used as cache key
    ContentHash  hash;          ///< hash of raw source at time of caching
 
    // ── Payload 1 : source ───────────────────────────────────────────────────
    std::optional<std::string>              source;      ///< raw UTF-8 source text
    std::optional<std::vector<io::SourceLoc>> span_index; ///< pre-built span table
 
    // ── Payload 2 : AST ──────────────────────────────────────────────────────
    std::shared_ptr<ASTNode>                ast_root;    ///< null if not yet parsed
 
    // ── Payload 3 : exports ──────────────────────────────────────────────────
    std::shared_ptr<ModuleExports>          exports;     ///< null if not yet resolved
 
    // ── Payload 4 : tensor metadata ──────────────────────────────────────────
    std::vector<TensorOpMeta>               tensor_ops;
 
    // ── Validity check ───────────────────────────────────────────────────────
 
    /// Returns true if `current_hash` matches the stored hash, meaning the
    /// cached data is still valid for the current source on disk.
    bool is_valid(const ContentHash& current_hash) const noexcept
    {
        return hash == current_hash;
    }
};
 
// ─── Thread-local cache ──────────────────────────────────────────────────────
 
/// Owned by a single compiler thread; no locking required.
/// Accumulates resolved modules during parallel import resolution.
/// Call `GlobalModuleCache::merge(*this)` when the thread's work is done.
class ThreadLocalCache
{
public:
    /// Insert or overwrite an entry.
    void put(const std::string& path, CachedModule entry)
    {
        entries_[path] = std::move(entry);
    }
 
    /// Look up an entry and validate its hash against `current_hash`.
    /// Returns nullptr if absent or stale.
    const CachedModule* get(const std::string& path,
                            const ContentHash&  current_hash) const
    {
        auto it = entries_.find(path);
        if (it == entries_.end()) return nullptr;
        return it->second.is_valid(current_hash) ? &it->second : nullptr;
    }
 
    /// Direct access to the underlying map (used by merge).
    const std::unordered_map<std::string, CachedModule>& entries() const
    {
        return entries_;
    }
 
    void clear() { entries_.clear(); }
 
private:
    std::unordered_map<std::string, CachedModule> entries_;
};
 
// ─── Global module cache ─────────────────────────────────────────────────────
 
/// Process-wide, thread-safe cache.
///
/// ### Typical workflow
/// ```
/// // In each worker thread:
/// ThreadLocalCache local;
/// local.put(path, build_module(path));
///
/// // When thread finishes:
/// GlobalModuleCache::instance().merge(local);
///
/// // In the main thread (or any thread) for lookup:
/// auto* m = GlobalModuleCache::instance().get(path, hash);
/// ```
class GlobalModuleCache
{
public:
    /// Meyer's singleton — one cache per process.
    static GlobalModuleCache& instance()
    {
        static GlobalModuleCache cache;
        return cache;
    }
 
    // Non-copyable, non-movable.
    GlobalModuleCache(const GlobalModuleCache&)            = delete;
    GlobalModuleCache& operator=(const GlobalModuleCache&) = delete;
 
    // ── Read ─────────────────────────────────────────────────────────────────
 
    /// Returns a copy of the cached entry if present and valid.
    /// Uses a shared lock — multiple threads can read concurrently.
    std::optional<CachedModule> get(const std::string& path,
                                    const ContentHash&  current_hash) const
    {
        std::shared_lock lock(mutex_);
        auto it = entries_.find(path);
        if (it == entries_.end()) return std::nullopt;
        if (!it->second.is_valid(current_hash)) return std::nullopt;
        return it->second;
    }
 
    /// Check existence without copying the payload.
    bool contains(const std::string& path,
                  const ContentHash& current_hash) const
    {
        std::shared_lock lock(mutex_);
        auto it = entries_.find(path);
        return it != entries_.end() && it->second.is_valid(current_hash);
    }
 
    // ── Write ─────────────────────────────────────────────────────────────────
 
    /// Merge all entries from a thread-local cache into the global cache.
    /// Entries already present are only overwritten if the incoming hash
    /// differs (i.e. the thread resolved a fresher version).
    void merge(const ThreadLocalCache& local)
    {
        std::unique_lock lock(mutex_);
        for (auto& [path, entry] : local.entries())
        {
            auto it = entries_.find(path);
            if (it == entries_.end() || !it->second.is_valid(entry.hash))
                entries_[path] = entry;
        }
    }
 
    /// Directly insert a single entry (e.g. from the main thread).
    void put(const std::string& path, CachedModule entry)
    {
        std::unique_lock lock(mutex_);
        entries_[path] = std::move(entry);
    }
 
    /// Evict a single entry (e.g. after a source file is modified on disk).
    void evict(const std::string& path)
    {
        std::unique_lock lock(mutex_);
        entries_.erase(path);
    }
 
    /// Drop all cached data (e.g. between incremental compilation sessions).
    void clear()
    {
        std::unique_lock lock(mutex_);
        entries_.clear();
    }
 
    /// Number of currently cached modules.
    size_t size() const
    {
        std::shared_lock lock(mutex_);
        return entries_.size();
    }
 
    /// Returns all cached paths (for diagnostics / debug).
    std::vector<std::string> cached_paths() const
    {
        std::shared_lock lock(mutex_);
        std::vector<std::string> out;
        out.reserve(entries_.size());
        for (auto& [p, _] : entries_) out.push_back(p);
        return out;
    }
 
private:
    GlobalModuleCache() = default;
 
    mutable std::shared_mutex                             mutex_;
    std::unordered_map<std::string, CachedModule>         entries_;
};
 
} // namespace io