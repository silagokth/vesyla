import os
import sys
import argparse
import logging

import generate
import schedule
import sync
import time

def main(args):
    # record the start time
    start_time = time.time()

    # define logging format
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    # parse the arguments
    parser = argparse.ArgumentParser(description='vs-schedule')
    parser.add_argument('-s', '--step', type=str, help='step to execute: [all, generate, schedule, sync]', default='all')
    parser.add_argument('-p', '--proto_asm', type=str, help='proto assembly file', required=False)
    parser.add_argument('-c', '--constraint', type=str, help='proto file', required=False)
    parser.add_argument('-m', '--model', type=str, help='timing model file', required=False)
    parser.add_argument('-t', '--timing', type=str, help='timing table file', required=False)
    parser.add_argument('-o', '--output', type=str, help='output directory', default='.')
    args = parser.parse_args(args)

    # if output directory does not exist, create it
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    if args.step == 'all':
        if args.proto_asm is None:
            logging.error("proto_asm is required")
            sys.exit(1)
        if args.constraint is None:
            logging.error("constraint is required")
            sys.exit(1)
        # check file exist
        if not os.path.exists(args.proto_asm):
            logging.error("proto_asm file does not exist")
            sys.exit(1)
        if not os.path.exists(args.constraint):
            logging.error("constraint file does not exist")
            sys.exit(1)
        generate.generate(args.proto_asm, args.constraint, args.output)
        schedule.schedule(os.path.join(args.output, "model.txt"), args.output)
        sync.sync(args.proto_asm, os.path.join(args.output, "timing_table.json"), args.output)
    elif args.step == 'generate':
        if args.proto_asm is None:
            logging.error("proto_asm is required")
            sys.exit(1)
        if args.constraint is None:
            logging.error("constraint is required")
            sys.exit(1)
        # check file exist
        if not os.path.exists(args.proto_asm):
            logging.error("proto_asm file does not exist")
            sys.exit(1)
        if not os.path.exists(args.constraint):
            logging.error("constraint file does not exist")
            sys.exit(1)

        generate.generate(args.proto_asm, args.constraint, args.output)
    elif args.step == 'schedule':
        if args.model is None:
            logging.error("model is required")
            sys.exit(1)
        # check file exist
        if not os.path.exists(args.model):
            logging.error("model file does not exist")
            sys.exit(1)

        schedule.schedule(args.model, args.output)
    elif args.step == 'sync':
        if args.proto_asm is None:
            logging.error("proto_asm is required")
            sys.exit(1)
        if args.timing is None:
            logging.error("timing is required")
            sys.exit(1)
        # check file exist
        if not os.path.exists(args.proto_asm):
            logging.error("proto_asm file does not exist")
            sys.exit(1)
        if not os.path.exists(args.timing):
            logging.error("timing file does not exist")
            sys.exit(1)
        sync.sync(args.proto_asm, args.timing, args.output)
    else:
        logging.error("Invalid step: "+args.step)
        sys.exit(1)
    
    # record the end time
    end_time = time.time()
    
    # write the execution time to the log file
    with open(os.path.join(args.output, "time.txt"), "w") as file:
        file.write(str(end_time-start_time)+"\n")

if __name__ == '__main__':
    main(sys.argv[1:])