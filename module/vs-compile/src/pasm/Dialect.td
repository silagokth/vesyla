#ifndef VESYLA_IROPT_PASM_DIALECT
#define VESYLA_IROPT_PASM_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Pasm_Dialect : Dialect {
    let name = "pasm";
    let summary = "A dialect to represent the control and data flow of a program with explicit indexing information.";
    let description = [{
        A dialect to represent the control and data flow of a program with explicit indexing information.
    }];
    let cppNamespace = "vesyla::pasm";
    let useDefaultTypePrinterParser = 1;
    let extraClassDeclaration = [{
    	void registerOps();
        void registerTypes();
    }];
}

#endif // VESYLA_IROPT_PASM_DIALECT
