#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import subprocess
from pathlib import Path
import glob

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

    os.system("vs-init -f -s " + style)
    os.system("cp -rf " + directory + "/* .")
    os.system("sh run.sh")

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

    print(f"In total, {len(testcases)} testcases are found:")
    for tc in testcases:
        print("\t" + tc.name)

    with open("autotest_config.robot", "w") as f:
        f.write("""
*** Settings ***
Library           Process
Library           OperatingSystem
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
    Empty Directory    work
    ${result} =    Run Process    vs-testcase run ${filename}    shell=True    timeout=30 min    stdout=stdout.txt    stderr=stderr.txt    cwd=work
    Should Be Equal As Integers    ${result.rc}    0
""")
    os.system("mkdir -p work")

def main():
    args = process_arguments()
    cmd = args.command
    arg = args.arguments
    if cmd == "run":
        run(arg[0])
    elif cmd == "generate":
        generate(arg[0])
    else:
        print("Unknown command")

if __name__ == "__main__":
    main()
