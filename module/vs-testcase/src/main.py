#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import subprocess
import glob
import sys
import logging
import argparse
import datetime

class Arguments:
    def __init__(self):
        self.command = os.sys.argv[1]
        self.arguments = os.sys.argv[2:]

def process_arguments():
    return Arguments()

def run(directory):
    path_config = os.path.join(directory, "arch.json")

    with open(path_config) as config_file:
        config = json.load(config_file)
    style = config["platform"]

    init(["-f", "-s", style, "-o", "."])
    rt = subprocess.run("cp -rf " + directory + "/* .", shell=True)
    if rt.returncode != 0:
        sys.exit(-1)
    rt = subprocess.run("sh run.sh", shell=True)
    if rt.returncode != 0:
        sys.exit(-1)

def generate(directory):
    pattern = os.path.join(directory, "*", "*", "*")
    leaf_dirs = [f for f in glob.glob(pattern) if os.path.isdir(f)]

    class TestcaseEntry:
        def __init__(self, name, path, tags):
            self.name = name
            self.path = path
            self.tags = tags

    testcases = []
    for dir in leaf_dirs:
        vec = str(dir).split(os.sep)
        name = "::".join(vec[-3:])
        path = str(dir)
        tags = "::".join(vec[-3:-1])
        testcases.append(TestcaseEntry(name, path, tags))

    logging.info(f"In total, {len(testcases)} testcases are found:")
    for tc in testcases:
        logging.info("\t" + tc.name)

    with open("autotest_config.robot", "w") as f:
        f.write("""
*** Settings ***
Library           Process
Library           OperatingSystem
Library           String
Suite Teardown    Terminate All Processes    kill=True
Test Template     Autotest Template

*** Test Cases ***  filename
""")
        for tc in testcases:
            f.write(f"tc {tc.name}    {tc.path}\n    [Tags]    {tc.tags}\n")

        f.write("""
*** Keywords ***
Autotest Template
    [Arguments]  ${filename}
    ${random_string} =    Generate Random String    12    [LOWER]
    Create Directory    work/${random_string}
    ${result} =    Run Process    vs-testcase run ${filename}    shell=True    timeout=30 min    stdout=stdout.txt    stderr=stderr.txt    cwd=work/${random_string}
    Should Be Equal As Integers    ${result.rc}    0
    Remove Directory   work/${random_string}     recursive=True
""")
    
    with open("run.sh", "w") as f:
        f.write("""
#!/bin/sh
pabot --testlevelsplit -d output autotest_config.robot
""")
    # add execute permission to run.sh
    rt = subprocess.run("chmod +x run.sh", shell=True)
    if rt.returncode != 0:
        sys.exit(-1)
    rt = subprocess.run("mkdir -p work", shell=True)
    if rt.returncode != 0:
        sys.exit(-1)

def init(args):
    # parse the arguments:
    # -s style, mandatory
    # -f force, optional, if set, overwrite the existing files
    # -o output directory

    parser = argparse.ArgumentParser(description="Initialize a new testcase")
    parser.add_argument("-s", "--style", help="style of the testcase", required=True)
    parser.add_argument("-f", "--force", help="force overwrite", action="store_true")
    parser.add_argument("-o", "--output", help="output directory", default=".")
    args = parser.parse_args(args)

    style = args.style
    force = args.force
    output = args.output

    # check if the output directory exists, if not, create it
    if not os.path.exists(output):
        os.makedirs(output)

    # check if the output directory has a ".lock" file, if so, it's locked.
    if os.path.exists(os.path.join(output, ".lock")):
        if not force:
            logging.error("The output directory is locked")
            sys.exit(-1)
    else:
        with open(os.path.join(output, ".lock"), "w") as f:
            # write the current time to the lock file
            now = datetime.datetime.now()
            f.write(now.strftime("%Y-%m-%d %H:%M:%S"))

    # get environment variable "VESYLA_SUITE_PATH_PROG"
    prog_path = os.getenv("VESYLA_SUITE_PATH_PROG")
    if not prog_path:
        logging.error("Environment variable VESYLA_SUITE_PATH_PROG is not set")
        sys.exit(-1)
    template_path = os.path.join(prog_path, "share/vesyla-suite/template", style)
    if not os.path.exists(template_path):
        logging.error(f"Template for style {style} does not exist")
        sys.exit(-1)
    # copy templates
    rt = subprocess.run(f"cp -rf {template_path}/* {output}", shell=True)
    if rt.returncode != 0:
        logging.error("Failed to copy templates")
        sys.exit(-1)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = process_arguments()
    cmd = args.command
    arg = args.arguments
    print(cmd)
    if cmd == "init":
        init(arg)
    elif cmd == "run":
        run(arg[0])
    elif cmd == "generate":
        generate(arg[0])
    else:
        logging.error("Unknown command")
        sys.exit(-1)

if __name__ == "__main__":
    main()
