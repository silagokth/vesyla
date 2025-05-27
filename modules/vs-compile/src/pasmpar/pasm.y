%{
    #include <stdio.h>
    #include <stdlib.h>
    #include "util/Common.hpp"
    #include "pasmpar/bison_util.hpp"

    #include <vector>
    #include <string>
    #include <unordered_map>
    #define LOOKAHEAD yychar
    #define YYDEBUG 1

    extern int yylex(void);
    static void yyerror(const char*);
    using namespace vesyla::pasmpar;

    struct operation_t {
        std::string kind;
        mlir::Operation* op;
    };
    struct operation_vector_t {
        std::vector<operation_t*> ops;
    };
    struct parameter_t {
        enum Kind{
            STRING,
            INT
        };
        Kind kind;
        std::string key;
        std::string val;
    };
    struct parameter_map_t {
        std::vector<parameter_t> params;
    };
%}

%locations


%union {
    char* strval;
    int intval;
    struct operation_t* op;
    struct operation_vector_t* op_vector;
    struct parameter_t* param;
    struct parameter_map_t* param_map;
}

%start PROG

%token <intval> INT
%token <strval> ID
%token FOR
%token IF
%token ELSE
%token EPOCH
%token ROP
%token COP
%token RAW
%token CONSTRAINT
%token <strval> STRING

%type <op> OP
%type <op_vector> OP_LIST
%type <param> PARAM
%type <param_map> PARAM_MAP
%type <op> ROP_OP
%type <op> COP_OP
%type <op> RAW_OP
%type <op> CONSTRAINT_OP
%type <op_vector> INSTR_LIST
%type <op> INSTR
%type <op_vector> REGION_LIST
%type <op> REGION
%type <op> EPOCH_REGION
%type <op> FOR_REGION
%type <op> IF_REGION


%%

PROG:
    REGION_LIST {
        LOG(DEBUG) << "PROG";
    }
    ;

REGION_LIST:
    REGION_LIST REGION {
        LOG(DEBUG) << "REGION_LIST";
        auto region_list = $1;
        region_list->ops.push_back($2);
        $$ = region_list;
    }
    | REGION {
        LOG(DEBUG) << "REGION_LIST";
        auto region_list = new operation_vector_t();
        region_list->ops.push_back($1);
        $$ = region_list;
    }
    ;

REGION:
    EPOCH_REGION {
        LOG(DEBUG) << "REGION";
        $$ = $1;
    }
    | FOR_REGION {
        LOG(DEBUG) << "REGION";
        $$ = $1;
    } 
    | IF_REGION {LOG(DEBUG) << "REGION";
        $$ = $1;} 
    ;

EPOCH_REGION:
    EPOCH '{' OP_LIST '}' {
        LOG(DEBUG) << "EPOCH_REGION";
        LOG(DEBUG) << "EPOCH_REGION";
        auto* op = new operation_t();
        op->kind = "EPOCH";
        if (auto epoch_op = llvm::dyn_cast<vesyla::pasm::EpochOp>(temp_epoch_op)) {
            // create a InstrOp
            mlir::OpBuilder builder(epoch_op.getBody());
            auto loc = builder.getUnknownLoc();
            builder.setInsertionPointToEnd(module->getBody());
            auto rop_op = builder.create<vesyla::pasm::EpochOp>(loc,
                builder.getStringAttr(vesyla::util::Common::gen_random_string(8)));
            mlir::Region& region = rop_op.getBody();
            region.push_back(new mlir::Block());
            builder.setInsertionPointToEnd(&region.back());
            for (auto&& instr : $3->ops) {
                // copy the instr to the new region
                auto instr_op = instr->op;
                auto new_instr_op = builder.clone(*instr_op);
                // remove the old instr_op
                instr_op->erase();
            }
            // Clean up instr_list
            delete $3;
            // add a yield operation
            builder.create<vesyla::pasm::YieldOp>(loc);
            op->op = rop_op.getOperation();
        }
        else {
            yyerror("EpochOp not found");
            exit(1);
        }
        $$ = op;
    }
        
    | EPOCH '<' ID '>' '{' OP_LIST '}' {
        LOG(DEBUG) << "EPOCH_REGION";
        auto* op = new operation_t();
        op->kind = "EPOCH";
        if (auto epoch_op = llvm::dyn_cast<vesyla::pasm::EpochOp>(temp_epoch_op)) {
            // create a InstrOp
            mlir::OpBuilder builder(epoch_op.getBody());
            auto loc = builder.getUnknownLoc();
            builder.setInsertionPointToEnd(module->getBody());
            auto rop_op = builder.create<vesyla::pasm::EpochOp>(loc,
                builder.getStringAttr($3));
            mlir::Region& region = rop_op.getBody();
            region.push_back(new mlir::Block());
            builder.setInsertionPointToEnd(&region.back());
            for (auto&& instr : $6->ops) {
                // copy the instr to the new region
                auto instr_op = instr->op;
                auto new_instr_op = builder.clone(*instr_op);
                // remove the old instr_op
                instr_op->erase();
            }
            // Clean up instr_list
            delete $6;
            // add a yield operation
            builder.create<vesyla::pasm::YieldOp>(loc);
            op->op = rop_op.getOperation();
        }
        else {
            yyerror("EpochOp not found");
            exit(1);
        }
        $$ = op;
    }
    ;

FOR_REGION:
    FOR '(' PARAM_MAP ')' '{' REGION_LIST '}' {LOG(DEBUG) << "FOR_REGION";}
    | FOR '<' ID '>' '(' PARAM_MAP ')' '{' REGION_LIST '}' {LOG(DEBUG) << "FOR_REGION";}
    ;

IF_REGION:
    IF '(' PARAM_MAP ')' '{' REGION_LIST '}' {LOG(DEBUG) << "IF_REGION";}
    | IF '<' ID '>' '(' PARAM_MAP ')' '{' REGION_LIST '}' {LOG(DEBUG) << "IF_REGION";}
    | IF '(' PARAM_MAP ')' '{' REGION_LIST '}' ELSE '{' REGION_LIST '}' {LOG(DEBUG) << "IF_REGION";}
    | IF '<' ID '>' '(' PARAM_MAP ')' '{' REGION_LIST '}' ELSE '{' REGION_LIST '}' {LOG(DEBUG) << "IF_REGION";}
    | IF '(' PARAM_MAP ')' '{' REGION_LIST '}' ELSE IF_REGION {LOG(DEBUG) << "IF_REGION";}
    | IF '<' ID '>' '(' PARAM_MAP ')' '{' REGION_LIST '}' ELSE IF_REGION {LOG(DEBUG) << "IF_REGION";}
    ;

PARAM_MAP:
    PARAM_MAP ',' PARAM    {
        LOG(DEBUG) << "PARAM_MAP";
        $1->params.push_back(*$3);
        $$ = $1;
        delete $3;
    }
    | PARAM {
        LOG(DEBUG) << "PARAM_MAP";
        $$ = new parameter_map_t();
        $$->params.push_back(*$1);
        delete $1;
    }
    ;

PARAM:
    ID '=' ID  {
        LOG(DEBUG) << "PARAM";
        $$ = new parameter_t();
        $$->kind = parameter_t::STRING;
        $$->key = $1;
        $$->val = $3;
    }
    | ID '=' INT  {
        LOG(DEBUG) << "PARAM";
        $$ = new parameter_t();
        $$->key = $1;
        $$->val = std::to_string($3);
        $$->kind = parameter_t::INT;
    }
    ;

OP_LIST:
    OP_LIST OP {LOG(DEBUG) << "OP_LIST"; $1->ops.push_back($2); $$ = $1;} 
    | OP {LOG(DEBUG) << "OP_LIST"; $$ = new operation_vector_t(); $$->ops.push_back($1);}
    ;
OP:
    ROP_OP {LOG(DEBUG) << "OP"; $$ = $1;}
    | COP_OP {LOG(DEBUG) << "OP";$$ = $1;}
    | RAW_OP {LOG(DEBUG) << "OP";$$ = $1; }
    | CONSTRAINT_OP {LOG(DEBUG) << "OP";$$ = $1;}
    ;

ROP_OP:
    ROP '(' PARAM_MAP ')' '{' INSTR_LIST '}' {
        LOG(DEBUG) << "ROP_OP";
        auto* op = new operation_t();
        op->kind = "ROP";
        if (auto epoch_op = llvm::dyn_cast<vesyla::pasm::EpochOp>(temp_epoch_op)) {
            // create a InstrOp
            mlir::OpBuilder builder(epoch_op.getBody());
            auto loc = builder.getUnknownLoc();
            auto param_map = $3;
            int row = -1;
            int col = -1;
            int slot = -1;
            int port = -1;
            if (param_map) {
                // Extract parameters from param_map
                for (auto& param : param_map->params) {
                    if (param.key == "row") {
                        row = std::stoi(param.val);
                    } else if (param.key == "col") {
                        col = std::stoi(param.val);
                    } else if (param.key == "slot") {
                        slot = std::stoi(param.val);
                    } else if (param.key == "port") {
                        port = std::stoi(param.val);
                    }
                }
            }
            if (row == -1 || col == -1 || slot == -1 || port == -1) {
                yyerror("Missing required parameters: row, col, slot, port");
                exit(1);
            }
            
            // Clean up param_map
            delete param_map;
            auto rop_op = builder.create<vesyla::pasm::RopOp>(loc,
                builder.getStringAttr(vesyla::util::Common::gen_random_string(8)),
                builder.getIntegerAttr(builder.getI32Type(), row),
                builder.getIntegerAttr(builder.getI32Type(), col),
                builder.getIntegerAttr(builder.getI32Type(), slot),
                builder.getIntegerAttr(builder.getI32Type(), port));
            mlir::Region& region = rop_op.getBody();
            region.push_back(new mlir::Block());
            builder.setInsertionPointToEnd(&region.back());
            for (auto&& instr : $6->ops) {
                // copy the instr to the new region
                auto instr_op = instr->op;
                auto new_instr_op = builder.clone(*instr_op);
                // remove the old instr_op
                instr_op->erase();
            }
            // Clean up instr_list
            delete $6;
            // add a yield operation
            builder.create<vesyla::pasm::YieldOp>(loc);
            op->op = rop_op.getOperation();
        }
        else {
            yyerror("EpochOp not found");
            exit(1);
        }

        $$ = op;
    }
    | ROP '<' ID '>' '(' PARAM_MAP ')' '{' INSTR_LIST '}' {
        LOG(DEBUG) << "ROP_OP";
        auto* op = new operation_t();
        op->kind = "ROP";
        if (auto epoch_op = llvm::dyn_cast<vesyla::pasm::EpochOp>(temp_epoch_op)) {
            // create a InstrOp
            mlir::OpBuilder builder(epoch_op.getBody());
            auto loc = builder.getUnknownLoc();
            auto param_map = $6;
            int row = -1;
            int col = -1;
            int slot = -1;
            int port = -1;
            if (param_map) {
                // Extract parameters from param_map
                for (auto& param : param_map->params) {
                    if (param.key == "row") {
                        row = std::stoi(param.val);
                    } else if (param.key == "col") {
                        col = std::stoi(param.val);
                    } else if (param.key == "slot") {
                        slot = std::stoi(param.val);
                    } else if (param.key == "port") {
                        port = std::stoi(param.val);
                    }
                }
            }
            if (row == -1 || col == -1 || slot == -1 || port == -1) {
                yyerror("Missing required parameters: row, col, slot, port");
                exit(1);
            }
            
            // Clean up param_map
            delete param_map;
            auto rop_op = builder.create<vesyla::pasm::RopOp>(loc,
                builder.getStringAttr($3),
                builder.getIntegerAttr(builder.getI32Type(), row),
                builder.getIntegerAttr(builder.getI32Type(), col),
                builder.getIntegerAttr(builder.getI32Type(), slot),
                builder.getIntegerAttr(builder.getI32Type(), port));
            mlir::Region& region = rop_op.getBody();
            region.push_back(new mlir::Block());
            builder.setInsertionPointToEnd(&region.back());
            for (auto&& instr : $9->ops) {
                // copy the instr to the new region
                auto instr_op = instr->op;
                auto new_instr_op = builder.clone(*instr_op);
                // remove the old instr_op
                instr_op->erase();
            }
            // Clean up instr_list
            delete $9;
            // add a yield operation
            builder.create<vesyla::pasm::YieldOp>(loc);
            op->op = rop_op.getOperation();
        }
        else {
            yyerror("EpochOp not found");
            exit(1);
        }
        $$ = op;
    }
    ;

COP_OP:
    COP '(' PARAM_MAP ')' '{' INSTR_LIST '}' {
        LOG(DEBUG) << "COP_OP";
        auto* op = new operation_t();
        op->kind = "COP";
        if (auto epoch_op = llvm::dyn_cast<vesyla::pasm::EpochOp>(temp_epoch_op)) {
            // create a InstrOp
            mlir::OpBuilder builder(epoch_op.getBody());
            auto loc = builder.getUnknownLoc();
            auto param_map = $3;
            int row = -1;
            int col = -1;
            if (param_map) {
                // Extract parameters from param_map
                for (auto& param : param_map->params) {
                    if (param.key == "row") {
                        row = std::stoi(param.val);
                    } else if (param.key == "col") {
                        col = std::stoi(param.val);
                    }
                }
            }
            if (row == -1 || col == -1) {
                yyerror("Missing required parameters: row, col");
                exit(1);
            }
            
            // Clean up param_map
            delete param_map;
            auto cop_op = builder.create<vesyla::pasm::CopOp>(loc,
                builder.getStringAttr(vesyla::util::Common::gen_random_string(8)),
                builder.getIntegerAttr(builder.getI32Type(), row),
                builder.getIntegerAttr(builder.getI32Type(), col));
            mlir::Region& region = cop_op.getBody();
            region.push_back(new mlir::Block());
            builder.setInsertionPointToEnd(&region.back());
            for (auto&& instr : $6->ops) {
                // copy the instr to the new region
                auto instr_op = instr->op;
                auto new_instr_op = builder.clone(*instr_op);
                // remove the old instr_op
                instr_op->erase();
            }
            // Clean up instr_list
            delete $6;

            op->op = cop_op.getOperation();
        }
        else {
            yyerror("EpochOp not found");
            exit(1);
        }
    }
    | COP '<' ID '>' '(' PARAM_MAP ')' '{' INSTR_LIST '}' {
        LOG(DEBUG) << "COP_OP";
        auto* op = new operation_t();
        op->kind = "COP";
        if (auto epoch_op = llvm::dyn_cast<vesyla::pasm::EpochOp>(temp_epoch_op)) {
            // create a InstrOp
            mlir::OpBuilder builder(epoch_op.getBody());
            auto loc = builder.getUnknownLoc();
            auto param_map = $6;
            int row = -1;
            int col = -1;
            if (param_map) {
                // Extract parameters from param_map
                for (auto& param : param_map->params) {
                    if (param.key == "row") {
                        row = std::stoi(param.val);
                    } else if (param.key == "col") {
                        col = std::stoi(param.val);
                    }
                }
            }
            if (row == -1 || col == -1) {
                yyerror("Missing required parameters: row, col");
                exit(1);
            }
            
            // Clean up param_map
            delete param_map;
            auto cop_op = builder.create<vesyla::pasm::CopOp>(loc,
                builder.getStringAttr($3),
                builder.getIntegerAttr(builder.getI32Type(), row),
                builder.getIntegerAttr(builder.getI32Type(), col));
            mlir::Region& region = cop_op.getBody();
            region.push_back(new mlir::Block());
            builder.setInsertionPointToEnd(&region.back());
            for (auto&& instr : $9->ops) {
                // copy the instr to the new region
                auto instr_op = instr->op;
                auto new_instr_op = builder.clone(*instr_op);
                // remove the old instr_op
                instr_op->erase();
            }
            // Clean up instr_list
            delete $9;

            op->op = cop_op.getOperation();
        }
        else {
            yyerror("EpochOp not found");
            exit(1);
        }
        $$ = op;
    }
    ;

RAW_OP:
    RAW '(' PARAM_MAP ')' '{' INSTR_LIST '}' {
        LOG(DEBUG) << "RAW_OP";
        auto* op = new operation_t();
        op->kind = "RAW";
        if (auto epoch_op = llvm::dyn_cast<vesyla::pasm::EpochOp>(temp_epoch_op)) {
            // create a InstrOp
            mlir::OpBuilder builder(epoch_op.getBody());
            auto loc = builder.getUnknownLoc();
            auto param_map = $3;
            int row = -1;
            int col = -1;
            if (param_map) {
                // Extract parameters from param_map
                for (auto& param : param_map->params) {
                    if (param.key == "row") {
                        row = std::stoi(param.val);
                    } else if (param.key == "col") {
                        col = std::stoi(param.val);
                    }
                }
            }
            if (row == -1 || col == -1) {
                yyerror("Missing required parameters: row, col");
                exit(1);
            }
            
            // Clean up param_map
            delete param_map;
            auto raw_op = builder.create<vesyla::pasm::RawOp>(loc,
                builder.getStringAttr(vesyla::util::Common::gen_random_string(8)),
                builder.getIntegerAttr(builder.getI32Type(), row),
                builder.getIntegerAttr(builder.getI32Type(), col));
            mlir::Region& region = raw_op.getBody();
            region.push_back(new mlir::Block());
            builder.setInsertionPointToEnd(&region.back());
            for (auto&& instr : $6->ops) {
                // copy the instr to the new region
                auto instr_op = instr->op;
                auto new_instr_op = builder.clone(*instr_op);
                // remove the old instr_op
                instr_op->erase();
            }
            // Clean up instr_list
            delete $6;
            // add a yield operation
            builder.create<vesyla::pasm::YieldOp>(loc);
            op->op = raw_op.getOperation();
        }
        else {
            yyerror("EpochOp not found");
            exit(1);
        }
        $$ = op;
    }
    | RAW '<' ID '>' '(' PARAM_MAP ')' '{' INSTR_LIST '}' {
        LOG(DEBUG) << "RAW_OP";
        auto* op = new operation_t();
        op->kind = "RAW";
        if (auto epoch_op = llvm::dyn_cast<vesyla::pasm::EpochOp>(temp_epoch_op)) {
            // create a InstrOp
            mlir::OpBuilder builder(epoch_op.getBody());
            auto loc = builder.getUnknownLoc();
            auto param_map = $6;
            int row = -1;
            int col = -1;
            if (param_map) {
                // Extract parameters from param_map
                for (auto& param : param_map->params) {
                    if (param.key == "row") {
                        row = std::stoi(param.val);
                    } else if (param.key == "col") {
                        col = std::stoi(param.val);
                    }
                }
            }
            if (row == -1 || col == -1) {
                yyerror("Missing required parameters: row, col, slot, port");
                exit(1);
            }
            
            // Clean up param_map
            delete param_map;
            auto raw_op = builder.create<vesyla::pasm::RawOp>(loc,
                builder.getStringAttr($3),
                builder.getIntegerAttr(builder.getI32Type(), row),
                builder.getIntegerAttr(builder.getI32Type(), col));
            mlir::Region& region = raw_op.getBody();
            region.push_back(new mlir::Block());
            builder.setInsertionPointToEnd(&region.back());
            for (auto&& instr : $9->ops) {
                // copy the instr to the new region
                auto instr_op = instr->op;
                auto new_instr_op = builder.clone(*instr_op);
                // remove the old instr_op
                instr_op->erase();
            }
            // Clean up instr_list
            delete $9;
            // add a yield operation
            builder.create<vesyla::pasm::YieldOp>(loc);
            op->op = raw_op.getOperation();
        }
        else {
            yyerror("EpochOp not found");
            exit(1);
        }
        $$ = op;
    }
    ;

CONSTRAINT_OP:
    CONSTRAINT '(' STRING ')' {
        LOG(DEBUG) << "CONSTRAINT_OP";
        auto* op = new operation_t();
        op->kind = "CONSTRAINT";
        if (auto epoch_op = llvm::dyn_cast<vesyla::pasm::EpochOp>(temp_epoch_op)) {
            // create a InstrOp
            mlir::OpBuilder builder(epoch_op.getBody());
            auto loc = builder.getUnknownLoc();
            std::string constraint_contents = $3;
            // remove quotes from constraint_contents
            if (constraint_contents.front() == '"' && constraint_contents.back() == '"') {
                constraint_contents = constraint_contents.substr(1, constraint_contents.size() - 2);
            }
            auto constraint_op = builder.create<vesyla::pasm::ConstraintOp>(loc,
                builder.getStringAttr("linear"),
                builder.getStringAttr(constraint_contents));
            op->op = constraint_op.getOperation();
        }
        else {
            yyerror("EpochOp not found");
            exit(1);
        }
        $$ = op;
    }
    ;

INSTR_LIST:
    INSTR_LIST INSTR {
        LOG(DEBUG) << "INSTR_LIST";
        $1->ops.push_back($2);
        $$ = $1;
    }
    | INSTR {
        LOG(DEBUG) << "INSTR_LIST";
        $$ = new operation_vector_t();
        $$->ops.push_back($1);
    }
    ;
INSTR:
    ID {
        LOG(DEBUG) << "INSTR";
        auto* op = new operation_t();
        op->kind = "INSTR";
        // Find a EpochOp in module named "__temp__"
        vesyla::pasm::EpochOp* epoch_op = nullptr;
        for (auto&& op : module->getOps<vesyla::pasm::EpochOp>()) {
            if (op.getId() == "__temp__") {
                // Found the EpochOp
                epoch_op = &op;
                break;
            }
        }
        if (epoch_op == nullptr) {
            yyerror("EpochOp not found");
            exit(1);
        }
        // create a InstrOp
        mlir::OpBuilder builder(epoch_op->getBody());
        auto loc = builder.getUnknownLoc();
        auto instrOp = builder.create<vesyla::pasm::InstrOp>(loc,
            builder.getStringAttr(vesyla::util::Common::gen_random_string(8)),
            builder.getStringAttr(std::string($1)),
            builder.getDictionaryAttr({}));
        op->op = instrOp.getOperation();
        $$ = op;
    }
    | ID '(' PARAM_MAP ')' {
        LOG(DEBUG) << "INSTR";
        auto* op = new operation_t();
        op->kind = "INSTR";
        // Find a EpochOp in module named "__temp__"
        vesyla::pasm::EpochOp* epoch_op = nullptr;
        for (auto&& op : module->getOps<vesyla::pasm::EpochOp>()) {
            if (op.getId() == "__temp__") {
                // Found the EpochOp
                epoch_op = &op;
                break;
            }
        }
        if (epoch_op == nullptr) {
            yyerror("EpochOp not found");
            exit(1);
        }
        // create a InstrOp
        mlir::OpBuilder builder(epoch_op->getBody());
        auto loc = builder.getUnknownLoc();
        auto param_map = $3;
        // Collect existing key-value pairs
        llvm::SmallVector<mlir::NamedAttribute, 8> attributes;

        // Add new key-value pairs from param_map
        for (auto& param : param_map->params) {
            auto key = builder.getStringAttr(param.key);
            if (param.kind == parameter_t::STRING) {
                // Handle string parameters
                auto val = builder.getStringAttr(param.val);
                attributes.push_back(builder.getNamedAttr(key, val));
            } else if (param.kind == parameter_t::INT) {
                // Handle integer parameters
                auto val = builder.getI32IntegerAttr(std::stoi(param.val));
                attributes.push_back(builder.getNamedAttr(key, val));
            }
        }

        // Create a new DictionaryAttr with the updated attributes
        auto dictionary_attr = builder.getDictionaryAttr(attributes);

        // Clean up param_map
        delete param_map;
        auto instrOp = builder.create<vesyla::pasm::InstrOp>(loc,
            builder.getStringAttr(vesyla::util::Common::gen_random_string(8)),
            builder.getStringAttr(std::string($1)),
            dictionary_attr);
        op->op = instrOp.getOperation();
        $$ = op;
    }
    | ID '<' ID '>' '(' PARAM_MAP ')' {
        LOG(DEBUG) << "INSTR";
        auto* op = new operation_t();
        op->kind = "INSTR";
        // Find a EpochOp in module named "__temp__"
        vesyla::pasm::EpochOp* epoch_op = nullptr;
        for (auto&& op : module->getOps<vesyla::pasm::EpochOp>()) {
            if (op.getId() == "__temp__") {
                // Found the EpochOp
                epoch_op = &op;
                break;
            }
        }
        if (epoch_op == nullptr) {
            yyerror("EpochOp not found");
            exit(1);
        }
        // create a InstrOp
        mlir::OpBuilder builder(epoch_op->getBody());
        auto loc = builder.getUnknownLoc();
        auto param_map = $6;
        // Collect existing key-value pairs
        llvm::SmallVector<mlir::NamedAttribute, 8> attributes;
        // Add new key-value pairs from param_map
        for (auto& param : param_map->params) {
            auto key = builder.getStringAttr(param.key);
            if (param.kind == parameter_t::STRING) {
                // Handle string parameters
                auto val = builder.getStringAttr(param.val);
                attributes.push_back(builder.getNamedAttr(key, val));
            } else if (param.kind == parameter_t::INT) {
                // Handle integer parameters
                auto val = builder.getI32IntegerAttr(std::stoi(param.val));
                attributes.push_back(builder.getNamedAttr(key, val));
            }
        }
        // Create a new DictionaryAttr with the updated attributes
        auto dictionary_attr = builder.getDictionaryAttr(attributes);
        // Clean up param_map
        delete param_map;
        auto instrOp = builder.create<vesyla::pasm::InstrOp>(loc,
            builder.getStringAttr($3),
            builder.getStringAttr(std::string($1)),
            dictionary_attr);
        op->op = instrOp.getOperation();
        $$ = op;
    }
    ;

%%

static void yyerror(const char* message)
{
  vesyla::pasmpar::print_error(message);
  exit(1);
}