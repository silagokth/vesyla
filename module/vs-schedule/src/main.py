import os
import sys
import argparse
import logging
import time
import dispatch

def main(args):
    # record the start time
    start_time = time.time()

    # define logging format
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    # parse the arguments
    parser = argparse.ArgumentParser(description='vs-schedule')
    parser.add_argument('-p', '--proto_asm', type=str, help='proto assembly file', required=False)
    parser.add_argument('-c', '--constraint', type=str, help='constraint file', required=False)
    parser.add_argument('-o', '--output', type=str, help='output directory', default='.')
    args = parser.parse_args(args)

    # if output directory does not exist, create it
    if not os.path.exists(args.output):
        os.makedirs(args.output)

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
    
    dispatch.dispatch(args.proto_asm, args.constraint, args.output)
    
    # record the end time
    end_time = time.time()
    
    # write the execution time to the log file
    with open(os.path.join(args.output, "time.txt"), "w") as file:
        file.write(str(end_time-start_time)+"\n")

if __name__ == '__main__':
    main(sys.argv[1:])