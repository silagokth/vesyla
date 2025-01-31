%start prog
%%
prog -> AsmProg :
    record_list {
        AsmProg { records: $1 } 
    }
    ;

record_list -> Vec<AsmRecord> :
    record_list record {
        let mut v = $1; v.push($2);
        v
    }
    | record {
        vec![$1]
    }
    ;

record -> AsmRecord :
    'IDENTIFIER' '(' parameter_map ')' 'ENDLINE' {
        let v1 = $1.map_err(|_| ()).unwrap();
        let sv1 = $lexer.span_str(v1.span());
        AsmRecord{id : get_random_id(), name : sv1.to_string(), parameters : $3}
    }
    | 'IDENTIFIER' '<' 'IDENTIFIER' '>' '(' parameter_map ')' 'ENDLINE' {
        let v1 = $1.map_err(|_| ()).unwrap();
        let sv1 = $lexer.span_str(v1.span());
        let v3 = $3.map_err(|_| ()).unwrap();
        let sv3 = $lexer.span_str(v3.span());
        AsmRecord{id : sv3.to_string(), name : sv1.to_string(), parameters : $6}
    }
    | 'IDENTIFIER' 'ENDLINE' {
        let v1 = $1.map_err(|_| ()).unwrap();
        let sv1 = $lexer.span_str(v1.span());
        AsmRecord{id : get_random_id(), name : sv1.to_string(), parameters : HashMap::new()}
    }
    | 'IDENTIFIER' '<' 'IDENTIFIER' '>' 'ENDLINE' {
        let v1 = $1.map_err(|_| ()).unwrap();
        let sv1 = $lexer.span_str(v1.span());
        let v3 = $3.map_err(|_| ()).unwrap();
        let sv3 = $lexer.span_str(v3.span());
        AsmRecord{id : sv3.to_string(), name : sv1.to_string(), parameters : HashMap::new()}
    }
    ;

parameter_map -> HashMap<String, String> :
    parameter_map ',' parameter {
        let mut m = $1; m.insert($3.0, $3.1);
        m
    }
    | parameter {
        let mut m = HashMap::new(); m.insert($1.0, $1.1);
        m
    }
    ;

parameter -> (String, String) :
    'IDENTIFIER' '=' 'IDENTIFIER' {
        let v1 = $1.map_err(|_| ()).unwrap();
        let sv1 = $lexer.span_str(v1.span());
        let v3 = $3.map_err(|_| ()).unwrap();
        let sv3 = $lexer.span_str(v3.span());
        (sv1.to_string(), sv3.to_string())
    }
    | 'IDENTIFIER' '=' 'NUMBER' {
        let v1 = $1.map_err(|_| ()).unwrap();
        let sv1 = $lexer.span_str(v1.span());
        let v3 = $3.map_err(|_| ()).unwrap();
        let sv3 = $lexer.span_str(v3.span());
        (sv1.to_string(), sv3.to_string())
    }
    ;
%%
use crate::asm::{AsmProg, AsmRecord};
use std::collections::HashMap;
use uuid::Uuid;

fn get_random_id() -> String {
        let uuid = Uuid::new_v4();
        let tag = uuid.to_string();
        let short_tag = "_".to_string() + &tag[..8].to_string();
        short_tag
    }
