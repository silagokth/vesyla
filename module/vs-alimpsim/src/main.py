#!/usr/bin/env python3

import Scheduler
import EventPool
import ResourcePool
import HandlerPool
import InitEvent
import os
import sys
import logging
import coloredlogs
import verboselogs
import json
import argparse

coloredlogs.install(datefmt='%H:%M:%S',
                    fmt='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)


def process_arguments():
    '''process command line arguments: --arch, --instr, --input, --output, --metric'''
    parser = argparse.ArgumentParser(description='Vesyla Simulator')
    parser.add_argument('--arch', metavar='arch', type=str, nargs=1,
                        help='architecture file')
    parser.add_argument('--isa', metavar='isa', type=str, nargs=1,
                        help='instruction set file')
    parser.add_argument('--instr', metavar='instr', type=str, nargs=1,
                        help='instruction file')
    parser.add_argument('--input', metavar='input', type=str, nargs=1,
                        help='input file')
    parser.add_argument('--output', metavar='output', type=str, nargs=1,
                        help='output file')
    parser.add_argument('--metric', metavar='metric', type=str, nargs=1,
                        help='metric file')
    # add an argument --step to indicate it's a step-by-step simulation
    parser.add_argument('--step', action='store_true',
                        help='step-by-step simulation')
    args = parser.parse_args()
    return args


def bindigits(n, bits):
    s = bin(n & int("1"*bits, 2))[2:]
    return ("{0:0>%s}" % (bits)).format(s)


if __name__ == "__main__":
    # parse arguments
    args = process_arguments()
    file_arch = args.arch[0]
    file_isa = args.isa[0]
    file_instr = args.instr[0]
    file_input = args.input[0]
    file_output = args.output[0]
    file_metric = args.metric[0]
    step = args.step

    ep = EventPool.EventPool()
    rp = ResourcePool.ResourcePool()
    hp = HandlerPool.HandlerPool()
    InitEvent.init_event(ep, rp, hp,
                         file_arch, file_isa, file_instr, file_input, file_output)
    sch = Scheduler.Scheduler(ep, rp, hp, step)
    sch.run()

    # Dump output buffer
    output_buffer = rp.get("output_buffer")
    output_buffer_active = rp.get("output_buffer_active")
    with open(file_output, "w+") as f:
        for addr in range(len(output_buffer_active)):
            if output_buffer_active[addr]:
                data = output_buffer[addr]
                print([addr, data])
                ss = ""
                for i in range(16):
                    ss += bindigits(data[16-i-1], 16)
                f.write(str(addr) + " " + ss + "\n")

    # Dump cost metrics
    iap = rp.get("IAP")
    oap = rp.get("OAP")
    metric = {
        "latency": sch.get_clk(),
        "area": 100,
        "energy": 100,
        "iap": iap,
        "oap": oap
    }
    with open(file_metric, "w+") as f:
        json.dump(metric, f)
