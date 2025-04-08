#ifndef VESYLA_IROPT_CIDFG_OPS
#define VESYLA_IROPT_CIDFG_OPS

include "cidfg/Dialect.td"
include "cidfg/Types.td"
include "cidfg/Passes.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinAttributes.td"


class Cidfg_Op<string mnemonic, list<Trait> traits = []> :
    Op<Cidfg_Dialect, mnemonic, traits>;

def IntegerConstantOp : Cidfg_Op<"IntegerConstant"> {
  let arguments = (ins I32Attr:$value);
  let results = (outs IntegerType: $result);
}

def CreateAffineIndexOp : Cidfg_Op<"AffineIndex"> {
  let arguments = (ins AnyTypeOf<[IntegerType, AffineIndexType]>:$init, IntegerType:$step, IntegerType:$iter);
  let results = (outs AffineIndexType: $result);
}

def AllocateOp : Cidfg_Op<"allocate"> {
  let arguments = (ins StrAttr:$bind, I32Attr:$chunk_num, I32Attr:$chunk_size);
  let results = (outs ArrayType: $result);
}

def ReadOp : Cidfg_Op<"read"> {
  let arguments = (ins ArrayType:$array, AffineIndexType:$index);
  let results = (outs StreamType: $result);
}

def WriteOp : Cidfg_Op<"write"> {
  let arguments = (ins ArrayType: $array, StreamType:$data, AffineIndexType:$index);
  let results = (outs ArrayType: $result);
}

def ScalarBinOp : Cidfg_Op<"scalar_binop"> {
  let arguments = (ins StrAttr:$op, IntegerType:$arg1, IntegerType:$arg2);
  let results = (outs IntegerType: $result);
}

def UnaryComputeOp : Cidfg_Op<"unary_compute"> {
  let arguments = (ins StrAttr:$op, StreamType:$arg1);
  let results = (outs StreamType: $result);
}

def BinaryComputeOp : Cidfg_Op<"binary_compute"> {
  let arguments = (ins StrAttr:$op, StreamType:$arg1, StreamType:$arg2);
  let results = (outs StreamType: $result);
}

def TenaryComputeOp : Cidfg_Op<"tenary_compute"> {
  let arguments = (ins StrAttr:$op, StreamType:$arg1, StreamType:$arg2, StreamType:$arg3);
  let results = (outs StreamType: $result);
}

#endif // VESYLA_IROPT_CIDFG_OPS
