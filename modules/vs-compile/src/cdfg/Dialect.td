#ifndef VESYLA_CDFG_DIALECT
#define VESYLA_CDFG_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def CdfgDialect : Dialect {
    let name = "cdfg";
    let summary = "A dialect to represent the control and data flow of a program with explicit indexing information.";
    let description = [{
        A dialect to represent the control and data flow of a program with explicit indexing information.
    }];
    let cppNamespace = "vesyla::cdfg";
    let useDefaultTypePrinterParser = 1;
    let extraClassDeclaration = [{
    	void registerOps();
        void registerTypes();
    }];
}

#endif // VESYLA_CDFG_DIALECT
