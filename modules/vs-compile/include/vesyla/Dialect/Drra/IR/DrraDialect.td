#ifndef VESYLA_DRRA_DIALECT
#define VESYLA_DRRA_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Drra_Dialect : Dialect {
    let name = "drra";
    let summary = "A dialect to represent DRRA resource operations.";
    let description = [{
        A dialect to represent DRRA resource operations.
    }];
    let cppNamespace = "vesyla::drra";
    let extraClassDeclaration = [{
        void registerOps();
    }];
}

#endif // VESYLA_DRRA_DIALECT
