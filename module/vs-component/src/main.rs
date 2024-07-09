use log::{info, warn, error, debug, trace};
use argparse;
use serde_json;
use minijinja;
use std::task::Context;

fn main(){
    // set the log level
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Debug).init();

    // parse command line arguments with the following format:
    // program_name [command] [options]
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        error!("Usage: {} [command] [options]", args[0]);
        std::process::exit(1);
    }

    // get the command
    let command = &args[1];
    let mut options : Vec<String> = Vec::new();
    for i in 1..args.len() {
        options.push(args[i].clone());
    }
    match command.as_str() {
        "gen_api" => {
            info!("Generating system verilog api ...");
            gen_api(options)
        },
        "gen_doc" => {
            info!("Generating markdown documentation ...");
            gen_doc(options)
        },
        _ => {
            error!("Unknown command: {}", command);
            panic!();
        }
    }

}

fn gen_api(args: Vec<String>) {
    // parse the "args" using argparse
    // -i <input file> -o <output file>
    let mut input_file = String::from("isa.json");
    let mut output_file = String::from("api.sv");
    {
        let mut ap = argparse::ArgumentParser::new();
        ap.set_description("Generate system verilog api");
        ap.refer(&mut input_file)
            .add_option(&["-i", "--input"], argparse::Store, "Input file");
        ap.refer(&mut output_file)
            .add_option(&["-o", "--output"], argparse::Store, "Output file");
        ap.parse(args, &mut std::io::stdout(), &mut std::io::stderr()).unwrap();
    }
    info!("Input file: {}", input_file);
    info!("Output file: {}", output_file);

    // read the json file
    let json_str = std::fs::read_to_string(input_file).expect("Failed to read file");
    let isa : serde_json::Value = serde_json::from_str(&json_str).expect("Failed to parse json");
    
    // using minijinja to generate the system verilog api
    let template = r#"
{%- set useful_bitwidth = isa.instr_bitwidth - isa.instr_code_bitwidth - 1 -%}
package {{isa.machine_type}}_{{isa.machine}}_api;
    {%- for instr in isa.instruction_templates %}
    // Instruction: {{instr.name}}
    opcode_{{instr.name}} = {{isa.instr_code_bitwidth}}'d{{instr.code}};
    typedef struct {
        {%- for field in instr.segment_templates %}
        {%- if field.bitwidth == 1 %}
        logic {{field.name}}; // {{field.comment}}
        {%- else %}
        logic[{{field.bitwidth - 1}}:0] {{field.name}}; // {{field.comment}}
        {%- endif %}
        {%- endfor %}
    }{{instr.name}}_t;
    function {{instr.name}}_t unpack_{{instr.name}};
        input logic [{{useful_bitwidth - 1}}:0] instr;
        {{instr.name}}_t st;
        {%- set start =useful_bitwidth - 1 %}
        {%- set end =0 %}
        {%- for field in instr.segment_templates %}
        {%- if field.bitwidth == 1 %}
        st.{{field.name}} = instr[{{start}}];
        {%- set start = start - 1 %}
        {%- else %}
        {%- set end = start - field.bitwidth + 1 %}
        st.{{field.name}} = instr[{{start}}:{{end}}];
        {%- set start = end - 1 %}
        {%- endif %}
        {%- endfor %}
        return st;
    endfunction
    function logic [{{useful_bitwidth - 1}}:0] pack_{{instr.name}};
        input {{instr.name}}_t st;
        logic [{{useful_bitwidth - 1}}:0] instr;
        {%- set start =useful_bitwidth-1 %}
        {%- set end =0 %}
        {%- for field in instr.segment_templates %}
        {%- if field.bitwidth == 1 %}
        instr[{{start}}] = st.{{field.name}};
        {%- set start = start - 1 %}
        {%- else %}
        {%- set end = start - field.bitwidth + 1 %}
        instr[{{start}}:{{end}}] = st.{{field.name}};
        {%- set start = end - 1 %}
        {%- endif %}
        {%- endfor %}
        return instr;
    endfunction
    {% endfor %}
endpackage
"#;

    let mut env = minijinja::Environment::new();
    env.add_template("api", template).unwrap();
    let tmpl = env.get_template("api").unwrap();
    let result =  tmpl.render(minijinja::context!(isa)).unwrap();

    // write the result to the output file
    std::fs::write(output_file, result).expect("Failed to write file");

}

fn gen_doc(args: Vec<String>) {
    // parse the "args" using argparse
    // -i <input file> -o <output file>
    let mut input_file = String::from("isa.json");
    let mut output_file = String::from("doc.md");
    {
        let mut ap = argparse::ArgumentParser::new();
        ap.set_description("Generate system verilog api");
        ap.refer(&mut input_file)
            .add_option(&["-i", "--input"], argparse::Store, "Input file");
        ap.refer(&mut output_file)
            .add_option(&["-o", "--output"], argparse::Store, "Output file");
        ap.parse(args, &mut std::io::stdout(), &mut std::io::stderr()).unwrap();
    }
    info!("Input file: {}", input_file);
    info!("Output file: {}", output_file);

    // read the json file
    let json_str = std::fs::read_to_string(input_file).expect("Failed to read file");
    let isa : serde_json::Value = serde_json::from_str(&json_str).expect("Failed to parse json");
    
    // using minijinja to generate the system verilog api
    let template = r#"
{%- set useful_bitwidth=isa.instr_bitwidth - isa.instr_code_bitwidth -%}
## {{isa.machine_type}} / {{isa.machine}}

{%- for i in isa.instruction_templates %}

### {{i.name}} [opcode={{i.code}}]

Field | Position | Width | Default Value | Description
------|----------|-------|---------------|-------------------------
{%- set start =useful_bitwidth-1 %}
{%- for j in i.segment_templates %}
{%- set end = start - j.bitwidth + 1 %}
{{j.name}} | [{{start }}, {{end}}] | {{j.bitwidth}} | {%- if j.default_val is defined -%}{{j.default_val}}{%- else -%}0{%- endif -%} | {{j.comment}} {%- if j.verbo_map -%}{% for k in j.verbo_map %} [{{k.key}}]:{{k.val}};{% endfor %}{%- endif -%}
{%- set start = end - 1 -%}
{%- endfor -%}
{%- endfor -%}
"#;

    let mut env = minijinja::Environment::new();
    env.add_template("doc", template).unwrap();
    let tmpl = env.get_template("doc").unwrap();
    let result =  tmpl.render(minijinja::context!(isa)).unwrap();

    // write the result to the output file
    std::fs::write(output_file, result).expect("Failed to write file");
 
    
}