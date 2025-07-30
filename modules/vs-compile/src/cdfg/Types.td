#ifndef VESYLA_CDFG_TYPE
#define VESYLA_CDFG_TYPE

include "cdfg/Dialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributes.td"

class CdfgType<string name, string typeMnemonic, list<Trait> traits = [],
               string baseCppClass = "::mlir::Type">
    : TypeDef<CdfgDialect, name, traits, baseCppClass> {
  let mnemonic = typeMnemonic;
}

def IntegerType : CdfgType<"Integer", "Integer"> {

  let description = [{
    An integer type.
  }];
  let parameters = (ins "unsigned": $width, StringRefParameter<"the custom value">: $bind);
  let assemblyFormat = "`<` $width `,`  $bind `>`";
}

def FloatType : CdfgType<"Float", "Float"> {
  let description = [{
    A float type.
  }];
  let parameters = (ins "unsigned": $width, "std::string": $bind);
  let assemblyFormat = "`<` $width `,`  $bind `>`";
}

def FixedType : CdfgType<"Fixed", "Fixed"> {
  let description = [{
    A fixed type.
  }];
  let parameters = (ins "unsigned": $int_width, "unsigned":$frac_width , "std::string": $bind);
  let assemblyFormat = "`<` $int_width `,`  $frac_width `,`  $bind `>`";
}

def ArrayType : CdfgType<"Array", "Array"> {

  let description = [{
    A bit vector type.
  }];
  let parameters = (ins "unsigned" : $chunk_num, "unsigned" : $chunk_size, "std::string": $bind);
  let assemblyFormat = "`<` $chunk_num `,` $chunk_size `,` $bind `>`";
}

def StreamType : CdfgType<"Stream", "Stream"> {

  let description = [{
    A stream type.
  }];
  let parameters = (ins "unsigned" : $chunk_num, "unsigned" : $chunk_size);
  let assemblyFormat = "`<` $chunk_num `,`  $chunk_size `>`";
}

def AffineIndexType : CdfgType<"AffineIndex", "AffineIndex"> {

  let description = [{
    An index type.
  }];
  let parameters = (ins "unsigned": $level);
  let assemblyFormat = "`<` $level `>`";
}

def IndexType : AnyTypeOf<[IntegerType, AffineIndexType]>;

#endif // VESYLA_CDFG_TYPE
