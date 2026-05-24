#include "module_handler.h"
#include "../ir/ir_modules/tensor_handler.h"
#include "../ir/ir_modules/std_handler.h"
#include "../ir/ir_modules/math_handler.h"
#include "../ir/ir_modules/nn_handler.h"
#include "../ir/ir_modules/optim_handler.h"
#include "../ir/ir_modules/data_handler.h"
#include "../ir/ir_modules/parallel_handler.h"
 
namespace io {
 
ModuleHandlerRegistry ModuleHandlerRegistry::with_builtins() {
    ModuleHandlerRegistry registry;
 
    // Core built-in handlers.
    // Each handler owns the lowering logic for one module namespace.
    // Adding a new module: implement ModuleHandler, register here.  Done.
    registry.register_handler(std::make_unique<ir::TensorModuleHandler>());
    registry.register_handler(std::make_unique<ir::StdModuleHandler>());
    registry.register_handler(std::make_unique<ir::MathModuleHandler>());
    registry.register_handler(std::make_unique<ir::NNModuleHandler>());
    registry.register_handler(std::make_unique<ir::OptimModuleHandler>());
    registry.register_handler(std::make_unique<ir::DataModuleHandler>());
    registry.register_handler(std::make_unique<ir::ParallelModuleHandler>());
 
    return registry;
}
 
} // namespace io