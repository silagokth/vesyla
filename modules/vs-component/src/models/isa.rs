use std::io::Error;

use serde::ser::{Serialize, SerializeMap, Serializer};

#[derive(Clone, Debug)]
pub struct InstructionSet {
    pub format: ISAFormat,
    pub component_instructions: Vec<ComponentInstructionSet>,
}

#[derive(Clone, Debug)]
pub struct ComponentInstructionSet {
    pub format: ISAFormat,
    pub instructions: Vec<Instruction>,
    pub component_type: Option<InstructionType>,
    pub component_kind: Option<String>,
}

#[derive(Clone, PartialEq, Debug)]
pub struct ISAFormat {
    pub instr_bitwidth: u8,
    pub type_bitwidth: u8,
    pub opcode_bitwidth: u8,
    pub slot_bitwidth: u8,
}

impl ISAFormat {
    pub fn new() -> Self {
        ISAFormat {
            instr_bitwidth: 0,
            type_bitwidth: 0,
            opcode_bitwidth: 0,
            slot_bitwidth: 0,
        }
    }

    pub fn is_none(&self) -> bool {
        self.instr_bitwidth == 0
            && self.type_bitwidth == 0
            && self.opcode_bitwidth == 0
            && self.slot_bitwidth == 0
    }
}

#[derive(Clone, Debug)]
pub struct Instruction {
    pub name: String,
    pub opcode: u8,
    pub instr_type: InstructionType,
    pub segments: Vec<Segment>,
}

#[derive(Clone, PartialEq, Debug)]
pub enum InstructionType {
    ControlInstruction = 0,
    ResourceInstruction = 1,
}

impl InstructionType {
    pub fn to_u8(&self) -> u8 {
        match self {
            InstructionType::ControlInstruction => 0,
            InstructionType::ResourceInstruction => 1,
        }
    }

    pub fn to_str(&self) -> &str {
        match self {
            InstructionType::ControlInstruction => "controller",
            InstructionType::ResourceInstruction => "resource",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Segment {
    pub name: String,
    pub comment: String,
    pub bitwidth: u8,
    pub verbo_map: Vec<VerboMapEntry>,
    pub default_val: Option<u64>,
    pub is_signed: bool,
}

#[derive(Clone, Debug)]
pub struct VerboMapEntry {
    pub key: u64,
    pub val: String,
}

impl Segment {
    pub fn from_json(segment_json: serde_json::Value) -> Result<Segment, Error> {
        let segment = segment_json;
        let name = segment.get("name").unwrap().as_str().unwrap().to_string();
        let comment = segment
            .get("comment")
            .unwrap()
            .as_str()
            .unwrap()
            .to_string();
        let bitwidth = segment.get("bitwidth").unwrap().as_u64().unwrap() as u8;

        // Parse verbo_map
        let mut verbo_map: Vec<VerboMapEntry> = Vec::new();
        let verbo_map_json = segment.get("verbo_map");
        if verbo_map_json.is_some() {
            verbo_map = VerboMapEntry::from_json(verbo_map_json.unwrap().clone())
                .expect("Failed to parse verbo_map");
        }

        let default_val = segment.get("default_val");
        let default_val = default_val.map(|default_val| default_val.as_u64().unwrap());
        let is_signed = segment.get("is_signed");
        let is_signed = match is_signed {
            Some(is_signed) => is_signed.as_bool().unwrap(),
            None => false,
        };

        Ok(Segment {
            name,
            comment,
            bitwidth,
            verbo_map,
            default_val,
            is_signed,
        })
    }
}

impl Instruction {
    pub fn from_json(instruction_json: serde_json::Value) -> Result<Instruction, Error> {
        let instruction = instruction_json;
        let name = instruction
            .get("name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                Error::new(
                    std::io::ErrorKind::InvalidData,
                    "Name not found or not a string",
                )
            })?
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
            _ => {
                return Err(Error::new(
                    std::io::ErrorKind::InvalidData,
                    "Invalid instruction type in ISA JSON",
                ))
            }
        };

        let mut segments = Vec::new();
        let segments_json = instruction.get("segments");
        if segments_json.is_none() {
            return Err(Error::new(
                std::io::ErrorKind::InvalidData,
                "Segments not found in ISA JSON",
            ));
        }
        if !segments_json.unwrap().is_array() {
            return Err(Error::new(
                std::io::ErrorKind::InvalidData,
                "Segments must be an array in ISA JSON",
            ));
        }
        for segment in segments_json.unwrap().as_array().unwrap().iter() {
            let segment_obj = Segment::from_json(segment.clone())?;
            segments.push(segment_obj);
        }

        Ok(Instruction {
            name,
            opcode,
            instr_type,
            segments,
        })
    }
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
    pub fn new() -> Self {
        InstructionSet {
            format: ISAFormat::new(),
            component_instructions: Vec::new(),
        }
    }

    pub fn add_format(&mut self, format: ISAFormat) -> Result<(), Error> {
        if self.format.is_none() {
            self.format = format;
        } else if self.format != format {
            return Err(Error::new(
                std::io::ErrorKind::InvalidData,
                "Incompatible ISA formats",
            ));
        }

        Ok(())
    }

    pub fn overwrite(&mut self, other: InstructionSet) -> Result<(), Error> {
        if !other.format.is_none() {
            self.format = other.format.clone();
        }
        if !other.component_instructions.is_empty() {
            self.add_component_isas(other.component_instructions.clone())?;
        }

        Ok(())
    }

    pub fn add_component_isa(
        &mut self,
        component_isa: ComponentInstructionSet,
    ) -> Result<(), Error> {
        if self.format.is_none() {
            self.format = component_isa.format.clone();
        } else if self.format != component_isa.format {
            return Err(Error::new(
                std::io::ErrorKind::InvalidData,
                "Incompatible ISA formats",
            ));
        }

        if self
            .component_instructions
            .iter()
            .any(|c| c.component_kind == component_isa.component_kind)
        {
            log::warn!(
                "Component kind {} already exists in ISA, skipping",
                component_isa
                    .component_kind
                    .as_ref()
                    .unwrap_or(&"unknown".to_string())
            );
        } else {
            self.component_instructions.push(component_isa);
        }

        Ok(())
    }

    pub fn add_component_isas(
        &mut self,
        component_isas: Vec<ComponentInstructionSet>,
    ) -> Result<(), Error> {
        for component_isa in component_isas.into_iter() {
            self.add_component_isa(component_isa)?;
        }
        Ok(())
    }
}

impl ComponentInstructionSet {
    pub fn from_json(isa_json: serde_json::Value) -> Result<ComponentInstructionSet, Error> {
        // Get format
        let format = isa_json.get("format");
        let format = match format {
            Some(format) => ISAFormat::from_json(format.clone())?,
            None => {
                return Err(Error::new(
                    std::io::ErrorKind::InvalidData,
                    "ISA format not found in ISA JSON",
                ))
            }
        };

        // Get kind
        let component_kind = isa_json
            .get("kind")
            .and_then(|k| k.as_str().map(|s| s.to_string()));
        if component_kind.is_none() {
            log::warn!("Component kind not found in ISA JSON");
        }

        // Get instructions and type
        let mut instr_type: Option<InstructionType> = None;
        let mut instructions = Vec::new();
        let instructions_json = isa_json.get("instructions");
        if instructions_json.is_none() {
            return Err(Error::new(
                std::io::ErrorKind::InvalidData,
                "Instructions not found in ISA JSON",
            ));
        }
        if !instructions_json.unwrap().is_array() {
            return Err(Error::new(
                std::io::ErrorKind::InvalidData,
                "Instructions must be an array in ISA JSON",
            ));
        }
        if instructions_json.unwrap().as_array().unwrap().is_empty() {
            return Err(Error::new(
                std::io::ErrorKind::InvalidData,
                "Instructions array is empty in ISA JSON",
            ));
        }

        let instructions_json = instructions_json.unwrap().as_array().unwrap();
        for instruction in instructions_json.iter() {
            let instruction_obj = Instruction::from_json(instruction.clone())?;
            if instr_type.is_none() {
                instr_type = Some(instruction_obj.instr_type.clone());
            }
            if instr_type != Some(instruction_obj.instr_type.clone()) {
                return Err(Error::new(
                    std::io::ErrorKind::InvalidData,
                    "Mixed instruction types in component ISA",
                ));
            }

            instructions.push(instruction_obj);
        }

        Ok(ComponentInstructionSet {
            format,
            instructions,
            component_type: instr_type,
            component_kind,
        })
    }

    pub fn overwrite(&mut self, other: ComponentInstructionSet) -> Result<(), Error> {
        if !other.format.is_none() {
            self.format = other.format.clone();
        }
        if !other.instructions.is_empty() {
            self.instructions = other.instructions.clone();
        }
        if other.component_type.is_some() {
            if self.component_type.is_some() && self.component_type != other.component_type {
                return Err(Error::new(
                    std::io::ErrorKind::InvalidData,
                    "Cannot overwrite component ISA with different instruction type",
                ));
            }
            self.component_type = other.component_type.clone();
        }
        if other.component_kind.is_some() {
            if self.component_kind.is_some() && self.component_kind != other.component_kind {
                return Err(Error::new(
                    std::io::ErrorKind::InvalidData,
                    "Cannot overwrite component ISA with different component kind",
                ));
            }
            self.component_kind = other.component_kind.clone();
        }

        Ok(())
    }

    pub fn validate(&self) -> Result<(), Error> {
        if self.format.is_none() {
            return Err(Error::new(
                std::io::ErrorKind::InvalidData,
                "ISA format not set in component ISA",
            ));
        }

        if self.component_kind.is_none() {
            return Err(Error::new(
                std::io::ErrorKind::InvalidData,
                "Component kind not set in component ISA",
            ));
        }
        if self.instructions.is_empty() {
            return Err(Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "No instructions in component {} ISA",
                    self.component_kind.as_ref().unwrap()
                ),
            ));
        }
        if self.component_type.is_none() {
            return Err(Error::new(
                std::io::ErrorKind::InvalidData,
                "Instruction type not set in component ISA",
            ));
        }

        Ok(())
    }
}

impl VerboMapEntry {
    pub fn from_json(verbo_map_json: serde_json::Value) -> Result<Vec<VerboMapEntry>, Error> {
        let mut verbo_map = Vec::new();
        for entry in verbo_map_json.as_array().unwrap().iter() {
            let key = entry.get("key").unwrap().as_u64().unwrap();
            let val = entry.get("val").unwrap().as_str().unwrap().to_string();
            verbo_map.push(VerboMapEntry { key, val });
        }
        Ok(verbo_map)
    }
}

impl Serialize for InstructionSet {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut map = serializer.serialize_map(Some(2))?;
        map.serialize_entry("format", &self.format)?;
        map.serialize_entry("components", &self.component_instructions)?;
        map.end()
    }
}

impl Serialize for ComponentInstructionSet {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut map = serializer.serialize_map(Some(4))?;
        map.serialize_entry(
            "component_type",
            &self.component_type.clone().unwrap().to_str(),
        )?;
        map.serialize_entry("kind", &self.component_kind)?;
        map.serialize_entry("instructions", &self.instructions)?;
        map.serialize_entry("format", &self.format)?;
        map.end()
    }
}

impl Serialize for Instruction {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut map = serializer.serialize_map(Some(4))?;
        map.serialize_entry("name", &self.name)?;
        map.serialize_entry("opcode", &self.opcode)?;
        map.serialize_entry("instr_type", &self.instr_type.to_u8())?;
        map.serialize_entry("segments", &self.segments)?;
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
        if !self.verbo_map.is_empty() {
            map.serialize_entry("verbo_map", &self.verbo_map)?;
        }
        if self.default_val.is_some() {
            map.serialize_entry("default_val", &self.default_val)?;
        }
        if self.is_signed {
            map.serialize_entry("is_signed", &self.is_signed)?;
        }
        map.end()
    }
}

impl Serialize for VerboMapEntry {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut map = serializer.serialize_map(Some(2))?;
        map.serialize_entry("key", &self.key)?;
        map.serialize_entry("val", &self.val)?;
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
