#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import argparse
import DataBase_pb2 as db
import Parser as par
import CodeGenerator as cg


def main():
    # parse command line arguments, all arguments are required:
    # -i/--instr <instruction file>
    # -s/--isa <instruction set file>
    # -o/--output <output directory>

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--instr", help="Instruction file", required=True)
    parser.add_argument(
        "-s", "--isa", help="Instruction set file", required=True)
    parser.add_argument(
        "-o", "--output", help="Output directory", required=True)
    args = parser.parse_args()

    # check if the instruction file exists
    if not os.path.isfile(args.instr):
        print("Error: Instruction file not found: {}".format(args.instr))
        exit(-1)
    # check if the instruction set file exists
    if not os.path.isfile(args.isa):
        print("Error: Instruction set file not found: {}".format(args.isa))
        exit(-1)
    # check if the output directory exists, if not create it
    if not os.path.isdir(args.output):
        os.mkdir(args.output)

    # parse the instruction set file
    data = db.DataBase()
    parser = par.Parser()
    if not parser.load_isa(args.isa, data):
        print("Error: Failed to load instruction set")
        exit(-1)
    # parse the instruction file
    if not parser.load_instr(args.instr, data):
        print("Error: Failed to parse instruction file")
        exit(-1)

    # generate files to the output directory
    generator = cg.CodeGenerator()
    generator.run(data, args.output)

    print("Instruction successfully generated to {}".format(args.output))


if __name__ == "__main__":
    main()
