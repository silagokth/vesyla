use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct AsmProg {
    pub records: Vec<AsmRecord>,
}

#[derive(Debug, Clone)]
pub struct AsmRecord {
    pub name: String,
    pub id: String,
    pub parameters: HashMap<String, String>,
}
