use std::collections::HashMap;

#[derive(Debug)]
pub struct AsmProg {
    pub records: Vec<AsmRecord>,
}

#[derive(Debug)]
pub struct AsmRecord {
    pub name: String,
    pub id: String,
    pub parameters: HashMap<String, String>,
}
