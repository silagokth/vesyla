#ifndef VESYLA_CDFG_OPS
#define VESYLA_CDFG_OPS

include "cdfg/Dialect.td"
include "cdfg/Types.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinAttributes.td"


class CdfgOp<string mnemonic, list<Trait> traits = []> :
    Op<CdfgDialect, mnemonic, traits>;

def IntegerConstantOp : CdfgOp<"IntegerConstant"> {
  let arguments = (ins I32Attr:$value);
  let results = (outs IntegerType: $result);
}

def CreateAffineIndexOp : CdfgOp<"AffineIndex"> {
  let arguments = (ins AnyTypeOf<[IntegerType, AffineIndexType]>:$init, IntegerType:$step, IntegerType:$iter);
  let results = (outs AffineIndexType: $result);
}

def AllocateOp : CdfgOp<"allocate"> {
  let arguments = (ins StrAttr:$bind, I32Attr:$chunk_num, I32Attr:$chunk_size);
  let results = (outs ArrayType: $result);
}

def ReadOp : CdfgOp<"read"> {
  let arguments = (ins ArrayType:$array, AffineIndexType:$index);
  let results = (outs StreamType: $result);
}

def WriteOp : CdfgOp<"write"> {
  let arguments = (ins ArrayType: $array, StreamType:$data, AffineIndexType:$index);
  let results = (outs ArrayType: $result);
}

def ScalarBinOp : CdfgOp<"scalar_binop"> {
  let arguments = (ins StrAttr:$op, IntegerType:$arg1, IntegerType:$arg2);
  let results = (outs IntegerType: $result);
}

def UnaryComputeOp : CdfgOp<"unary_compute"> {
  let arguments = (ins StrAttr:$op, StreamType:$arg1);
  let results = (outs StreamType: $result);
}

def BinaryComputeOp : CdfgOp<"binary_compute"> {
  let arguments = (ins StrAttr:$op, StreamType:$arg1, StreamType:$arg2);
  let results = (outs StreamType: $result);
}

def TenaryComputeOp : CdfgOp<"tenary_compute"> {
  let arguments = (ins StrAttr:$op, StreamType:$arg1, StreamType:$arg2, StreamType:$arg3);
  let results = (outs StreamType: $result);
}

#endif // VESYLA_CDFG_OPS
