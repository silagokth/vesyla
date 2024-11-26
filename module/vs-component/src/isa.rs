use serde::ser::{Serialize, SerializeMap, Serializer};
use std::io::Error;

#[derive(Clone)]
pub struct InstructionSet {
    pub format: ISAFormat,
    pub instructions: Vec<Instruction>,
}

#[derive(Clone)]
pub struct ISAFormat {
    pub instr_bitwidth: u8,
    pub type_bitwidth: u8,
    pub opcode_bitwidth: u8,
    pub slot_bitwidth: u8,
}

#[derive(Clone)]
pub struct Instruction {
    pub name: String,
    pub opcode: u8,
    pub instr_type: InstructionType,
    pub segments: Vec<Segment>,
}

#[derive(Clone)]
pub enum InstructionType {
    ControlInstruction = 0,
    ResourceInstruction = 1,
}

#[derive(Clone)]
pub struct Segment {
    pub name: String,
    pub comment: String,
    pub bitwidth: u8,
}

impl ISAFormat {
    pub fn from_json(format_json: serde_json::Value) -> Result<ISAFormat, Error> {
        let instr_bitwidth = format_json.get("instr_bitwidth");
        let instr_bitwidth = match instr_bitwidth {
            Some(instr_bitwidth) => instr_bitwidth.as_u64().unwrap() as u8,
            None => Err(Error::new(
                std::io::ErrorKind::InvalidData,
                "Instruction bitwidth not found in ISA format JSON",
            ))?,
        };
        let type_bitwidth = format_json.get("instr_type_bitwidth");
        let type_bitwidth = match type_bitwidth {
            Some(type_bitwidth) => type_bitwidth.as_u64().unwrap() as u8,
            None => Err(Error::new(
                std::io::ErrorKind::InvalidData,
                "Instruction type bitwidth not found in ISA format JSON",
            ))?,
        };
        let opcode_bitwidth = format_json.get("instr_opcode_bitwidth");
        let opcode_bitwidth = match opcode_bitwidth {
            Some(opcode_bitwidth) => opcode_bitwidth.as_u64().unwrap() as u8,
            None => Err(Error::new(
                std::io::ErrorKind::InvalidData,
                "Opcode bitwidth not found in ISA format JSON",
            ))?,
        };
        let slot_bitwidth = format_json.get("instr_slot_bitwidth");
        let slot_bitwidth = match slot_bitwidth {
            Some(slot_bitwidth) => slot_bitwidth.as_u64().unwrap() as u8,
            None => Err(Error::new(
                std::io::ErrorKind::InvalidData,
                "Slot bitwidth not found in ISA format JSON",
            ))?,
        };
        Ok(ISAFormat {
            instr_bitwidth,
            type_bitwidth,
            opcode_bitwidth,
            slot_bitwidth,
        })
    }
}

impl InstructionSet {
    pub fn from_json(isa_json: serde_json::Value) -> Result<InstructionSet, Error> {
        let format = isa_json.get("format");
        let format = match format {
            Some(format) => ISAFormat::from_json(format.clone())?,
            None => Err(Error::new(
                std::io::ErrorKind::InvalidData,
                "Format not found in ISA JSON",
            ))?,
        };
        let mut instructions = Vec::new();
        let instructions_json = isa_json.get("instructions");
        if instructions_json.is_none() {
            return Err(Error::new(
                std::io::ErrorKind::InvalidData,
                "Instructions not found in ISA JSON",
            ));
        } else if !instructions_json.unwrap().is_array() {
            return Err(Error::new(
                std::io::ErrorKind::InvalidData,
                "Instructions must be an array in ISA JSON",
            ));
        } else if !instructions_json.unwrap().as_array().unwrap().is_empty() {
            for instruction in isa_json
                .get("instructions")
                .unwrap()
                .as_array()
                .unwrap()
                .iter()
            {
                let name = instruction
                    .get("name")
                    .unwrap()
                    .as_str()
                    .unwrap()
                    .to_string();
                let opcode = instruction.get("opcode");
                let opcode = match opcode {
                    Some(opcode) => opcode.as_u64().unwrap() as u8,
                    None => {
                        println!("{:?}", instruction);
                        return Err(Error::new(
                            std::io::ErrorKind::InvalidData,
                            "Opcode not found in ISA JSON",
                        ));
                    }
                };
                let instr_type = match instruction.get("instr_type").unwrap().as_u64().unwrap() {
                    0 => InstructionType::ControlInstruction,
                    1 => InstructionType::ResourceInstruction,
                    _ => panic!("Invalid instruction type"),
                };
                let mut segments = Vec::new();
                let segments_json = instruction.get("segments");
                if segments_json.is_none() {
                    panic!("Segments not found in ISA JSON");
                } else if !segments_json.unwrap().is_array() {
                    return Err(Error::new(
                        std::io::ErrorKind::InvalidData,
                        "Segments must be an array in ISA JSON",
                    ));
                } else if !segments_json.unwrap().as_array().unwrap().is_empty() {
                    for segment in segments_json.unwrap().as_array().unwrap().iter() {
                        let seg_name = segment.get("name").unwrap().as_str().unwrap().to_string();
                        let comment = segment
                            .get("comment")
                            .unwrap()
                            .as_str()
                            .unwrap()
                            .to_string();
                        let seg_bitwidth = segment.get("bitwidth").unwrap().as_u64().unwrap() as u8;
                        segments.push(Segment {
                            name: seg_name,
                            comment,
                            bitwidth: seg_bitwidth,
                        });
                    }
                    instructions.push(Instruction {
                        name,
                        opcode,
                        instr_type,
                        segments,
                    });
                }
            }
        }
        Ok(InstructionSet {
            format,
            instructions,
        })
    }
}

impl Serialize for InstructionSet {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut map = serializer.serialize_map(Some(5))?;
        map.serialize_entry("format", &self.format)?;
        map.serialize_entry("instructions", &self.instructions)?;
        map.end()
    }
}

impl Serialize for Instruction {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut map = serializer.serialize_map(Some(3))?;
        map.serialize_entry("name", &self.name)?;
        map.serialize_entry("opcode", &self.opcode)?;
        map.serialize_entry("type", &self.instr_type)?;
        map.serialize_entry("segments", &self.segments)?;
        map.end()
    }
}

impl Serialize for InstructionType {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut map = serializer.serialize_map(Some(1))?;
        match self {
            InstructionType::ControlInstruction => {
                map.serialize_entry("type", &0)?;
            }
            InstructionType::ResourceInstruction => {
                map.serialize_entry("type", &1)?;
            }
        }
        map.end()
    }
}

impl Serialize for Segment {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut map = serializer.serialize_map(Some(3))?;
        map.serialize_entry("name", &self.name)?;
        map.serialize_entry("comment", &self.comment)?;
        map.serialize_entry("bitwidth", &self.bitwidth)?;
        map.end()
    }
}

impl Serialize for ISAFormat {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut map = serializer.serialize_map(Some(4))?;
        map.serialize_entry("instr_bitwidth", &self.instr_bitwidth)?;
        map.serialize_entry("instr_type_bitwidth", &self.type_bitwidth)?;
        map.serialize_entry("instr_opcode_bitwidth", &self.opcode_bitwidth)?;
        map.serialize_entry("instr_slot_bitwidth", &self.slot_bitwidth)?;
        map.end()
    }
}
