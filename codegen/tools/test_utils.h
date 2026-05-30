/// codegen/tools/test_utils.h
///
/// Utilities for testing code generation:
/// - Assembly parsing and validation
/// - Output comparison against reference/expected patterns
/// - Common test infrastructure for legacy and progressive systems

#pragma once

#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <regex>
#include <cassert>
#include <iostream>

namespace codegen::testing {

/// Represents an assembly instruction parsed from output
struct AsmInstr {
    std::string label;    // e.g., "entry_block:", empty if not a label
    std::string mnemonic; // e.g., "add", "ld", "ret"
    std::vector<std::string> operands; // e.g., {"a0", "a1", "a2"}
    
    bool is_label() const { return !label.empty(); }
};

/// Parse assembly file into structured instruction list
inline std::vector<AsmInstr> parse_assembly(const std::string& asm_text)
{
    std::vector<AsmInstr> instrs;
    std::istringstream iss(asm_text);
    std::string line;
    
    while (std::getline(iss, line)) {
        // Skip empty lines and directives
        if (line.empty() || line[0] == '.' || line[0] == '#')
            continue;
        
        // Trim leading/trailing whitespace
        size_t start = line.find_first_not_of(" \t");
        if (start == std::string::npos) continue;
        line = line.substr(start);
        
        AsmInstr instr;
        
        // Check for label (ends with ':')
        if (line.back() == ':' || line.find(':') != std::string::npos) {
            size_t colon_pos = line.find(':');
            instr.label = line.substr(0, colon_pos);
            instrs.push_back(instr);
            continue;
        }
        
        // Parse mnemonic and operands
        std::istringstream iss_instr(line);
        iss_instr >> instr.mnemonic;
        
        // Remove comment if present
        if (instr.mnemonic.find('#') != std::string::npos) {
            instr.mnemonic = instr.mnemonic.substr(0, instr.mnemonic.find('#'));
        }
        
        // Parse operands
        std::string operand_line;
        std::getline(iss_instr, operand_line);
        if (!operand_line.empty()) {
            // Simple tokenization by comma
            size_t pos = 0;
            while (pos < operand_line.length()) {
                size_t comma_pos = operand_line.find(',', pos);
                if (comma_pos == std::string::npos) comma_pos = operand_line.length();
                
                std::string op = operand_line.substr(pos, comma_pos - pos);
                // Trim whitespace
                size_t op_start = op.find_first_not_of(" \t");
                size_t op_end = op.find_last_not_of(" \t");
                if (op_start != std::string::npos) {
                    op = op.substr(op_start, op_end - op_start + 1);
                    if (!op.empty() && op[0] != '#')
                        instr.operands.push_back(op);
                }
                
                pos = comma_pos + 1;
            }
        }
        
        if (!instr.mnemonic.empty())
            instrs.push_back(instr);
    }
    
    return instrs;
}

/// Check if assembly contains required instructions
struct AssemblyValidator {
    std::vector<AsmInstr> instrs;
    std::string diagnostic;
    
    explicit AssemblyValidator(const std::string& asm_text)
        : instrs(parse_assembly(asm_text)) {}
    
    /// Check that mnemonic appears N times
    bool has_mnemonic(const std::string& mnemonic, int min_count = 1) const
    {
        int count = 0;
        for (const auto& instr : instrs) {
            if (instr.mnemonic == mnemonic)
                ++count;
        }
        return count >= min_count;
    }
    
    /// Check that specific instruction sequence exists
    bool has_pattern(const std::vector<std::string>& mnemonics) const
    {
        if (mnemonics.empty()) return true;
        
        for (size_t i = 0; i + mnemonics.size() <= instrs.size(); ++i) {
            bool match = true;
            for (size_t j = 0; j < mnemonics.size(); ++j) {
                if (instrs[i + j].mnemonic != mnemonics[j]) {
                    match = false;
                    break;
                }
            }
            if (match) return true;
        }
        return false;
    }
    
    /// Check that function ends with return-like instruction
    bool ends_with_return() const
    {
        for (auto it = instrs.rbegin(); it != instrs.rend(); ++it) {
            if (it->is_label()) continue;
            if (it->mnemonic.empty()) continue;
            
            // Valid return mnemonics for RISC-V, x86, ARM
            return it->mnemonic == "ret" ||    // x86 / RISC-V
                   it->mnemonic == "jr" ||     // RISC-V alternate
                   it->mnemonic == "bx";       // ARM
        }
        return false;
    }
    
    /// Get total number of instructions (excluding labels, directives)
    int instr_count() const
    {
        int count = 0;
        for (const auto& instr : instrs) {
            if (!instr.is_label() && !instr.mnemonic.empty())
                ++count;
        }
        return count;
    }
};

/// Read assembly file and validate it
inline std::string read_asm_file(const std::string& filename)
{
    std::ifstream ifs(filename);
    if (!ifs) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    std::stringstream buffer;
    buffer << ifs.rdbuf();
    return buffer.str();
}

/// Write reference assembly for manual inspection
inline void write_reference(const std::string& filename, const std::string& content)
{
    std::ofstream ofs(filename);
    if (ofs) {
        ofs << content;
    }
}

} // namespace codegen::testing
