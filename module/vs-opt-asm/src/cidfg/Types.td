#ifndef VESYLA_IROPT_CIDFG_TYPE
#define VESYLA_IROPT_CIDFG_TYPE

include "cidfg/Dialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributes.td"

class Cidfg_Type<string name, string typeMnemonic, list<Trait> traits = [],
               string baseCppClass = "::mlir::Type">
    : TypeDef<Cidfg_Dialect, name, traits, baseCppClass> {
  let mnemonic = typeMnemonic;
}

def IntegerType : Cidfg_Type<"Integer", "Integer"> {

  let description = [{
    An integer type.
  }];
  let parameters = (ins "unsigned": $width, StringRefParameter<"the custom value">: $bind);
  let assemblyFormat = "`<` $width `,`  $bind `>`";
}

def FloatType : Cidfg_Type<"Float", "Float"> {
  let description = [{
    A float type.
  }];
  let parameters = (ins "unsigned": $width, "std::string": $bind);
  let assemblyFormat = "`<` $width `,`  $bind `>`";
}

def FixedType : Cidfg_Type<"Fixed", "Fixed"> {
  let description = [{
    A fixed type.
  }];
  let parameters = (ins "unsigned": $int_width, "unsigned":$frac_width , "std::string": $bind);
  let assemblyFormat = "`<` $int_width `,`  $frac_width `,`  $bind `>`";
}

def ArrayType : Cidfg_Type<"Array", "Array"> {

  let description = [{
    A bit vector type.
  }];
  let parameters = (ins "unsigned" : $chunk_num, "unsigned" : $chunk_size, "std::string": $bind);
  let assemblyFormat = "`<` $chunk_num `,` $chunk_size `,` $bind `>`";
}

def StreamType : Cidfg_Type<"Stream", "Stream"> {

  let description = [{
    A stream type.
  }];
  let parameters = (ins "unsigned" : $chunk_num, "unsigned" : $chunk_size);
  let assemblyFormat = "`<` $chunk_num `,`  $chunk_size `>`";
}

def AffineIndexType : Cidfg_Type<"AffineIndex", "AffineIndex"> {

  let description = [{
    An index type.
  }];
  let parameters = (ins "unsigned": $level);
  let assemblyFormat = "`<` $level `>`";
}

def IndexType : AnyTypeOf<[IntegerType, AffineIndexType]>;

#endif // VESYLA_IROPT_CIDFG_TYPE
