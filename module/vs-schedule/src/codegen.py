import os
import sys
import json
import ds_pb2 as ds
from google.protobuf.json_format import MessageToJson
import logging

def id_to_text(id: str) -> str:
    if id == "" or id.startswith("__"):
        return ""
    return f"<{id}>"
def parameters_to_text(parameters) -> str:
    if len(parameters) == 0:
        return ""
    vp = []
    for parameter in parameters:
        vp.append(f"{parameter}={parameters[parameter]}")
    text = "(" + ", ".join(vp) + ")"
    return text

def pasmrecord_to_text(record_id: str, prog: ds.PASMProg) -> str:
    text = ""
    record = prog.records[record_id]
    if record.kind == ds.PASMRecord.Kind.START:
        for content in record.contents:
            text += pasmrecord_to_text(content, prog)
    elif record.kind == ds.PASMRecord.Kind.LOOP or record.kind == ds.PASMRecord.Kind.COND or record.kind == ds.PASMRecord.Kind.EPOCH or record.kind == ds.PASMRecord.Kind.CELL or record.kind == ds.PASMRecord.Kind.COP or record.kind == ds.PASMRecord.Kind.ROP:
        if record.kind == ds.PASMRecord.Kind.LOOP:
            label = "loop"
        elif record.kind == ds.PASMRecord.Kind.COND:
            label = "cond"
        elif record.kind == ds.PASMRecord.Kind.EPOCH:
            label = "epoch"
        elif record.kind == ds.PASMRecord.Kind.CELL:
            label = "cell"
        elif record.kind == ds.PASMRecord.Kind.COP:
            label = "cop"
        elif record.kind == ds.PASMRecord.Kind.ROP:
            label = "rop"
        elif record.kind == ds.PASMRecord.Kind.RAW:
            label = "raw"
        text += f"{label} {id_to_text(record.id)} {parameters_to_text(record.parameters)} {{\n"
        for content in record.contents:
            text += pasmrecord_to_text(content, prog)
        text += "}\n"
    elif record.kind == ds.PASMRecord.Kind.INSTR:
        text += f"{record.name} {id_to_text(record.id)} {parameters_to_text(record.parameters)}\n"
    else:
        logging.error(f"Unknown PASM record kind: {record.kind}")
        sys.exit(1)
    return text

def cstrrecord_to_text(record_id: str, prog: ds.CSTRProg) -> str:
    text = ""
    record = prog.records[record_id]
    if record.kind == ds.CSTRRecord.Kind.START:
        for content in record.contents:
            text += cstrrecord_to_text(content, prog)
    elif record.kind == ds.CSTRRecord.Kind.EPOCH:
        label = "epoch"
        text += f"{label} {id_to_text(record.id)} {parameters_to_text(record.parameters)} {{\n"
        for content in record.contents:
            text += cstrrecord_to_text(content, prog)
        text += "}\n"
    elif record.kind == ds.CSTRRecord.Kind.CSTR:
        text += f"{record.name} {record.constraint}\n"
    else:
        logging.error(f"Unknown CSTR record kind: {record.kind}")
        sys.exit(1)
    return text

def asmrecord_to_text(prog: ds.ASMProg) -> str:
    text = ""
    for content in prog.contents:
        record = prog.records[content]
        if record.kind == ds.ASMRecord.Kind.CELL:
            text += f"cell {id_to_text(record.id)} {parameters_to_text(record.parameters)} \n"
        elif record.kind == ds.ASMRecord.Kind.INSTR:
            text += f"{record.name} {id_to_text(record.id)} {parameters_to_text(record.parameters)}\n"
        else:
            logging.error(f"Unknown ASM record kind: {record.kind}")
            sys.exit(1)
    return text

def pasmprog_to_text(prog: ds.PASMProg) -> str:
    text = pasmrecord_to_text(prog.start, prog)
    return text

def pasmprog_to_json(prog: ds.PASMProg) -> str:
    json_text = MessageToJson(prog)
    return json_text

def cstrprog_to_text(prog: ds.CSTRProg) -> str:
    text = cstrrecord_to_text(prog.start, prog)
    return text

def cstrprog_to_json(prog: ds.CSTRProg) -> str:
    json_text = MessageToJson(prog)
    return json_text

def asmprog_to_text(prog: ds.ASMProg) -> str:
    text = asmrecord_to_text(prog)
    return text

def asmprog_to_json(prog: ds.ASMProg) -> str:
    json_text = MessageToJson(prog)
    return json_text



def parameters_to_pure_text(parameters) -> str:
    if len(parameters) == 0:
        return ""
    vp = []
    for parameter in parameters:
        vp.append(f"{parameter}={parameters[parameter]}")
    text = ", ".join(vp)
    return text

def pasmrecord_to_text(id, prog):
    text = ""
    if id in prog.records:
        record = prog.records[id]
        if record.kind == ds.PASMRecord.Kind.EPOCH:
            for content in record.contents:
                text += pasmrecord_to_text(content, prog)
        elif record.kind == ds.PASMRecord.Kind.CELL:
            x = record.parameters["x"]
            y = record.parameters["y"]
            label = f"{x}_{y}"
            text += f"cell {label}\n"
            for content in record.contents:
                text += pasmrecord_to_text(content, prog)
        elif record.kind == ds.PASMRecord.Kind.COP:
            text += f"cop {record.id} {parameters_to_pure_text(record.parameters)}\n"
            for content in record.contents:
                text += pasmrecord_to_text(content, prog)
        elif record.kind == ds.PASMRecord.Kind.ROP:
            text += f"rop {record.id} {parameters_to_pure_text(record.parameters)}\n"
            for content in record.contents:
                text += pasmrecord_to_text(content, prog)
        elif record.kind == ds.PASMRecord.Kind.INSTR:
            text += f"{record.name} {parameters_to_pure_text(record.parameters)}\n"
        else:
            logging.error(f"Unknown PASM record kind: {record.kind}")
            sys.exit(1)

    return text

def cstrrecord_to_text(id, prog):
    text = ""
    if id in prog.records:
        record = prog.records[id]
        if record.kind == ds.CSTRRecord.Kind.EPOCH:
            for content in record.contents:
                text += cstrrecord_to_text(content, prog)
        elif record.kind == ds.CSTRRecord.Kind.CSTR:
            text += f"type {record.name}\n"
            text += record.constraint + "\n"
        else:
            logging.error(f"Unknown CSTR record kind: {record.kind}")
            sys.exit(1)
    return text