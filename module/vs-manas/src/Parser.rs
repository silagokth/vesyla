use uuid::Uuid;

fn get_random_id -> string {
    let id = Uuid::new_v4();
    // add prefix of underscore
    "_".to_string() + id.to_string()
}

struct Parser {
}

impl Parser {
    fn new() -> Parser {
        Parser {}
    }

    fn convert_bin (s: string) -> int {
        let num = i32.from_str_radix(s, 2).unwrap();
        num
    }
    fn convert_oct (s: string) -> int {
        let num = i32.from_str_radix(s, 8).unwrap();
        num
    }
    fn convert_hex (s: string) -> int {
        let num = i32.from_str_radix(s, 16).unwrap();
        num
    }
    fn convert_dec (s: string) -> int {
        s.parse::<i32>().unwrap()
    }
    fn convert_num (s: string) -> int {
        if s.starts_with("0b") {
            convert_bin(s[2..])
        } else if s.starts_with("0x") {
            convert_hex(s[2..])
        } else if s.starts_with("0o") {
            convert_oct(s[2..])
        } else {
            convert_dec(s)
        }
    }

    fn load_isa (isa_file : string) -> InstructionSet {
        let mut isa = InstructionSet {
            platform: "".to_string(),
            format: HashMap::new(),
            instruction_templates: HashMap::new(),
        };
        let mut file = File::open(isa_file).unwrap();
        // read the isa file as json
        let mut contents = String::new();
        file.read_to_string(&mut contents).unwrap();
        let j: Value = serde_json::from_str(&contents).unwrap();
        isa.platform = j["platform"].as_str().unwrap().to_string();
        let format = j["format"].as_object().unwrap();

    }
}