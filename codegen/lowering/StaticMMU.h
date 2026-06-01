#pragma once
 
/// codegen/lowering/StaticMMU.h
///
/// StaticMMU — Compile-Time Scratchpad Memory Mapping (Phase B-2)
///
/// Hardware layout (Tensor-V, active 8 KB bank):
///
///   [0x0000 .. 0x17FF]  Input / Weight tile pool   6 KB  → 96 × 64 B tiles
///   [0x1800 .. 0x1FFF]  Accumulator / partial-sum  2 KB  →  8 × 256 B tiles
///
/// Each 8×8 INT8 tile    = 64 bytes  (one 512-bit cache line).
/// Each 8×8 FP32 partial = 256 bytes (partial-sum accumulator).
///
/// The pass:
///   1. Validates that each scratchpad buffer fits its partition.
///   2. Assigns final byte-precise bank offsets.
///   3. Tags each MemoryBuffer so the emitter can embed literal addresses
///      inside .insn DMA_LOAD / DMA_STORE / AME_COMPUTE directives.
///   4. Marks the LoopNest as requiring_spilling when either partition
///      overflows, so KReductionHandler can act.
 
#include "LoopNest.h"
#include "ScratchpadAllocator.h"
#include <cstdint>
#include <string>
#include <unordered_map>
#include <iostream>
 
namespace codegen::lowering {
 
// ── Hardware memory-map constants (from Tensor-V Project Proposal) ───────────
 
struct HWMemMap {
    static constexpr int64_t BANK_SIZE        = 8  * 1024;   // 8 KB
    static constexpr int64_t INPUT_BASE       = 0x0000;
    static constexpr int64_t INPUT_CAPACITY   = 6  * 1024;   // 6 KB
    static constexpr int64_t INPUT_TILE_BYTES = 64;           // 8×8 INT8
    static constexpr int64_t MAX_INPUT_TILES  = INPUT_CAPACITY / INPUT_TILE_BYTES; // 96
    static constexpr int64_t ACCUM_BASE       = 0x1800;
    static constexpr int64_t ACCUM_CAPACITY   = 2  * 1024;   // 2 KB
    static constexpr int64_t ACCUM_TILE_BYTES = 256;          // 8×8 FP32
    static constexpr int64_t MAX_ACCUM_TILES  = ACCUM_CAPACITY / ACCUM_TILE_BYTES; // 8
};
 
// ── Per-buffer resolved address ───────────────────────────────────────────────
 
struct MappedAddress {
    int64_t bank_offset = -1;   // byte offset within the active bank
    bool    is_accum    = false;
    bool    valid       = false;
};
 
// ── StaticMMU ─────────────────────────────────────────────────────────────────
 
class StaticMMU {
public:
    StaticMMU() = default;
 
    /// Run mapping over all scratchpad-resident buffers in llir.
    /// Must be called after ScratchpadAllocator::allocate().
    void map(LoopNest& llir, const ScratchpadAllocator& alloc) {
        address_map_.clear();
        int64_t input_cursor = HWMemMap::INPUT_BASE;
        int64_t accum_cursor = HWMemMap::ACCUM_BASE;
        input_used_ = 0;
        accum_used_ = 0;
 
        for (auto& buf : llir.buffers) {
            if (!buf->is_scratchpad) continue;
 
            MappedAddress addr;
            bool is_accum = (buf->role == BufferRole::Accumulator ||
                             buf->role == BufferRole::Temporary);
 
            if (is_accum) {
                if (accum_used_ + buf->total_bytes > HWMemMap::ACCUM_CAPACITY) {
                    std::cerr << "[StaticMMU] OVERFLOW: accum buffer '" << buf->name
                              << "' (" << buf->total_bytes << " B). Flagging spill.\n";
                    llir.requires_spilling = true;
                }
                addr.bank_offset = accum_cursor;
                addr.is_accum    = true;
                addr.valid       = true;
                accum_cursor += buf->total_bytes;
                accum_used_  += buf->total_bytes;
            } else {
                if (input_used_ + buf->total_bytes > HWMemMap::INPUT_CAPACITY) {
                    std::cerr << "[StaticMMU] OVERFLOW: input buffer '" << buf->name
                              << "' (" << buf->total_bytes << " B). Flagging spill.\n";
                    llir.requires_spilling = true;
                }
                addr.bank_offset = input_cursor;
                addr.is_accum    = false;
                addr.valid       = true;
                input_cursor += buf->total_bytes;
                input_used_  += buf->total_bytes;
            }
 
            address_map_[buf->name] = addr;
 
            if (verbose_) {
                std::cerr << "[StaticMMU] " << buf->name
                          << " @ 0x" << std::hex << addr.bank_offset << std::dec
                          << " (" << buf->total_bytes << " B, "
                          << (addr.is_accum ? "accum" : "input") << " partition)\n";
            }
        }
    }
 
    int64_t resolved_offset(const std::string& name) const {
        auto it = address_map_.find(name);
        return (it != address_map_.end() && it->second.valid)
               ? it->second.bank_offset : -1;
    }
 
    bool is_accum(const std::string& name) const {
        auto it = address_map_.find(name);
        return (it != address_map_.end()) && it->second.is_accum;
    }
 
    int64_t input_bytes_used() const { return input_used_; }
    int64_t accum_bytes_used() const { return accum_used_; }
 
    void set_verbose(bool v) { verbose_ = v; }
 
private:
    std::unordered_map<std::string, MappedAddress> address_map_;
    int64_t input_used_ = 0;
    int64_t accum_used_ = 0;
    bool    verbose_    = false;
};
 
} // namespace codegen::lowering