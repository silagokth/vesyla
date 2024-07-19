#![allow(unused_imports)]

use log::{debug, error, info, trace, warn};
use minijinja;
use serde_json;
use std::fs;

pub fn gen_doc(input_file: String, output_file: String) {
    // read the json file
    let json_str = std::fs::read_to_string(input_file).expect("Failed to read file");
    let isa: serde_json::Value = serde_json::from_str(&json_str).expect("Failed to parse json");

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
    let result = tmpl.render(minijinja::context!(isa)).unwrap();

    // write the result to the output file
    std::fs::write(output_file, result).expect("Failed to write file");
}

pub fn gen_api(input_file: String, output_file: String) {
    // read the json file
    let json_str = std::fs::read_to_string(input_file).expect("Failed to read file");
    let isa: serde_json::Value = serde_json::from_str(&json_str).expect("Failed to parse json");

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
    let result: String = tmpl.render(minijinja::context!(isa)).unwrap();

    // write the result to the output file
    std::fs::write(output_file, result).expect("Failed to write file");
}
