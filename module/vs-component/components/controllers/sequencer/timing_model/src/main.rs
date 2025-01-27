/**
 * This is the main file for the Rust implementation of the timing model extraction.
 * It requires two arguments: the input file and the output file.
 * The input file is a JSON file that contains the following fields:
 * - row: the number of rows in the timing model
 * - col: the number of columns in the timing model
 * - slot: the number of slots in the timing model
 * - port: the number of ports in the timing model
 * - op_name: the name of the operation
 * - instr_list: a list of instructions, each instruction contains the following fields:
 *  - name: the name of the instruction
 *  - fields: a list of fields, each field contains the following fields:
 *   - name: the name of the field
 *   - value: the value of the field
 * The output file is a text file that contains the timing model.
 *
 * The function extract_op_expr(op_name, instr_list) is the main function that
 * you need to implement. Don't change the function interface and any other functions.
 * You can add helper functions if needed.
 *
 */
use serde_json::Value;
use std::collections::HashMap;
use std::env;
use uuid::Uuid;

/*******************************************************************************
 * Modify here to implement the function. Don't change the function interface.
 ******************************************************************************/
fn extract_op_expr(op_name: String, instr_list: Vec<(String, Vec<(String, String)>)>) -> String {
    let mut t: HashMap<i64, String> = HashMap::new();
    let mut r: HashMap<i64, (String, String)> = HashMap::new();
    let mut expr = "".to_string();

    let counter = 0;
    for instr in instr_list {
        if counter == 0 {
            expr = format!("{}_e0", op_name);
        } else {
            expr = format!("T<{}>({}, {}_e{})", get_uuid_tag(), expr, op_name, counter);
        }
        counter += 1;
    }
    expr
}

/*******************************************************************************
 * Modification ends here
 ******************************************************************************/

fn main() {
    let (row, col, slot, port, op_name, instr_list, output_file) = parse_input();
    let timing_model = create_timing_model(row, col, slot, port, op_name, instr_list);
    std::fs::write(output_file, timing_model).unwrap();
}

fn parse_input() -> (
    i64,
    i64,
    i64,
    i64,
    String,
    Vec<(String, Vec<(String, String)>)>,
    String,
) {
    let args = env::args().collect::<Vec<String>>();
    if args.len() < 3 {
        println!("Usage: {} <input_file> <output_file>", args[0]);
        panic!("No input/output file specified!");
    }
    let input_file = &args[1]; // input_file
    let output_file = &args[2]; // output_file
                                // read the input file as json
    let input = std::fs::read_to_string(input_file).unwrap();
    let v: Value = serde_json::from_str(&input).unwrap();
    let row = v["row"].as_i64().unwrap();
    let col = v["col"].as_i64().unwrap();
    let slot = v["slot"].as_i64().unwrap();
    let port = v["port"].as_i64().unwrap();
    let op_name = v["op_name"].as_str().unwrap().to_string();
    let mut instr_list = Vec::new();
    for instr_json in v["instr_list"].as_array().unwrap() {
        let instr_name = instr_json["name"].as_str().unwrap().to_string();
        let mut instr_fields = Vec::new();
        for field_json in instr_json["fields"].as_array().unwrap() {
            let field_name = field_json["name"].as_str().unwrap().to_string();
            let field_value = field_json["value"].as_str().unwrap().to_string();
            instr_fields.push((field_name, field_value));
        }
        instr_list.push((instr_name, instr_fields));
    }
    (
        row,
        col,
        slot,
        port,
        op_name,
        instr_list,
        output_file.to_string(),
    )
}

fn create_timing_model(
    row: i64,
    col: i64,
    slot: i64,
    port: i64,
    op_name: String,
    instr_list: Vec<(String, Vec<(String, String)>)>,
) -> String {
    if row < 0 || col < 0 || slot < 0 || port < 0 {
        panic!("Invalid input");
    }
    let timing_model = extract_op_expr(op_name, instr_list);

    timing_model.to_string()
}

// Get an 8-character unique tag, starting with "__" and ending with "__"
fn get_uuid_tag() -> String {
    let uuid = Uuid::new_v4();
    let uuid_str = uuid.to_simple().to_string();
    let uuid_str = uuid_str.chars().take(8).collect::<String>();
    format!("__{}__", uuid_str)
}
