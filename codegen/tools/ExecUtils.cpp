#include "ExecUtils.h"
#include <fstream>
#include <cstdlib>
#include <windows.h>

namespace codegen::testing {

void* assemble_and_load(const std::string& asm_text, const std::string& work_dir_prefix) {
    std::string s_path = work_dir_prefix + ".s";
    std::string dll_path = work_dir_prefix + ".dll";
    std::string log_path = work_dir_prefix + ".log";
    {
        std::ofstream f(s_path);
        f << asm_text;
    }
    std::string cmd = "gcc -shared -o \"" + dll_path + "\" \"" + s_path + "\" > \"" + log_path + "\" 2>&1";
    int rc = std::system(cmd.c_str());
    if (rc != 0) return nullptr;
    return static_cast<void*>(LoadLibraryA(dll_path.c_str()));
}

void* get_symbol(void* module, const std::string& name) {
    if (!module) return nullptr;
    return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(module), name.c_str()));
}

void unload(void* module) {
    if (module) FreeLibrary(static_cast<HMODULE>(module));
}

} // namespace codegen::testing
