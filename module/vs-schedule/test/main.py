#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import re
import logging
import os
import random
import time

def create_constraint(anchor_list):
    if len(anchor_list) == 0:
        logging.error('Empty anchor list!')
        sys.exit(1)
    elif len(anchor_list) == 1:
        comparator = random.choice(['==', '<=', '>='])
        if comparator == '==':
            constraint = anchor_list[0][0] + ' == ' + str(anchor_list[0][1])
        elif comparator == '<=':
            constraint = anchor_list[0][0] + ' <= ' + str(anchor_list[0][1] + int(abs(random.random()*anchor_list[0][1])))
        elif comparator == '>=':
            constraint = anchor_list[0][0] + ' >= ' + str(anchor_list[0][1] - int(abs(random.random()*anchor_list[0][1])))
        return constraint
    elif len(anchor_list) == 2:
        comparator = random.choice(['==', '<=', '>='])
        distance = anchor_list[1][1] - anchor_list[0][1]
        if comparator == '==':
            constraint = anchor_list[1][0] + ' - ' + anchor_list[0][0] + ' == (' + str(distance) + ')'
        elif comparator == '<=':
            constraint = anchor_list[1][0] + ' - ' + anchor_list[0][0] + ' <= (' + str(distance + int(abs(random.random()*distance))) + ')'
        elif comparator == '>=':
            constraint = anchor_list[1][0] + ' - ' + anchor_list[0][0] + ' >= (' + str(distance - int(abs(random.random()*distance))) + ')'
        return constraint
    elif len(anchor_list) == 3:
        comparator = random.choice(['==', '<=', '>='])
        distance1 = anchor_list[1][1] - anchor_list[0][1]
        distance2 = anchor_list[2][1] - anchor_list[1][1]
        distance = distance1 - distance2
        if comparator == '==':
            constraint = anchor_list[1][0] + ' - ' + anchor_list[0][0] + ' == ' + anchor_list[2][0] + ' - ' + anchor_list[1][0] + ' + (' + str(distance) + ')'
        elif comparator == '<=':
            constraint = anchor_list[1][0] + ' - ' + anchor_list[0][0] + ' <= ' + anchor_list[2][0] + ' - ' + anchor_list[1][0] + ' + (' + str(distance + int(abs(random.random()*distance))) + ')'
        elif comparator == '>=':
            constraint = anchor_list[1][0] + ' - ' + anchor_list[0][0] + ' >= ' + anchor_list[2][0] + ' - ' + anchor_list[1][0] + ' + (' + str(distance - int(abs(random.random()*distance))) + ')'
        return constraint
    elif len(anchor_list) == 4:
        comparator = random.choice(['==', '<=', '>='])
        distance1 = anchor_list[1][1] - anchor_list[0][1]
        distance2 = anchor_list[3][1] - anchor_list[2][1]
        distance = distance1 - distance2
        if comparator == '==':
            constraint = anchor_list[1][0] + ' - ' + anchor_list[0][0] + ' == ' + anchor_list[3][0] + ' - ' + anchor_list[2][0] + ' + (' + str(distance) + ')'
        elif comparator == '<=':
            constraint = anchor_list[1][0] + ' - ' + anchor_list[0][0] + ' <= ' + anchor_list[3][0] + ' - ' + anchor_list[2][0] + ' + (' + str(distance + int(abs(random.random()*distance))) + ')'
        elif comparator == '>=':
            constraint = anchor_list[1][0] + ' - ' + anchor_list[0][0] + ' >= ' + anchor_list[3][0] + ' - ' + anchor_list[2][0] + ' + (' + str(distance - int(abs(random.random()*distance))) + ')'
        return constraint
    else:
        logging.error("Too many anchors in the anchor list!")
        sys.exit(1)

def create_operation(op_name, M, MAX_LATENCY) -> dict:
    op = {}
    op['name']=op_name
    offset = random.randint(0, int(0.5*MAX_LATENCY))
    expr = op_name+"_e0"
    event_counter = 1
    anchor_list = [[op_name+"_e0", "", 0]]
    curr_duration = 1
    for i in range(M):
        # randomly choose 'T' or 'R'
        op_type = random.choice(['T', 'R'])
        if op_type == 'R':
            delay = random.randint(2, 10)
            iter = random.randint(2, 10)
            new_expr = 'R<%d,%d>(%s)' % (iter, delay, expr)

            new_anchor_list = []
            for x in anchor_list:
                for i in range(iter):
                    new_anchor_list.append([x[0], '['+str(i)+']'+x[1], x[2]+i*curr_duration+i*delay])
            new_duration = iter*curr_duration + (iter-1)*delay
            if new_duration < MAX_LATENCY:
                # commit changes
                anchor_list = new_anchor_list
                curr_duration = new_duration
                expr = new_expr
            else:
                # try T operator
                op_type = 'T'
        if op_type == 'T':
            delay = random.randint(2, 10)
            new_event = op_name+'_e%d' % event_counter
            new_event_counter = event_counter + 1
            new_expr = 'T<%d>(%s, %s)' % (delay, expr, new_event)
            new_anchor_list = []
            new_anchor_list.append([new_event, "", curr_duration+delay])
            new_duration = curr_duration + (delay + 1)
            if new_duration < MAX_LATENCY:
                # commit changes
                anchor_list.extend(new_anchor_list)
                curr_duration = new_duration
                expr = new_expr
                event_counter = new_event_counter
            else:
                break
    op['expr'] = expr
    op['anchor_list'] = [[x[0]+x[1], x[2]+offset] for x in anchor_list]
    op['latency'] = curr_duration
    return op

def convert_anchor_to_anchor_name(anchor):
    anchor = anchor.strip()
    anchor.replace('\s', '')
    pattern = re.compile(r'([a-zA-Z_$][\w]*_e[0-9]+)([\[\d\]]*)')
    match = pattern.match(anchor)
    if match is None:
        logging.error('Invalid anchor format: %s' % anchor)
        sys.exit(1)
    event = match.group(1)
    index = match.group(2)
    if index is not None:
        index = index[1:-1]
        index = index.split('][')
        if len(index) == 1 and index[0] == '':
            index = []
    
    name = event
    if index != []:
        name = name + '_' + '_'.join(index)
    return name

def create_test_case(filename, N, M, C):
    MAX_LATENCY = 10000
    op_list = []
    anchor_list = []
    for i in range(N):
        logging.debug('Creating operation %d' % i)
        op = create_operation('op%d' % i, M, MAX_LATENCY)
        # randomly sample maximum 10 anchors from op['anchor_list']
        if len(op['anchor_list']) > 10:
            op['anchor_list'] = random.sample(op['anchor_list'], 10)
        for i in range(len(op['anchor_list'])):
            op['anchor_list'][i] = [op['anchor_list'][i][0], op['anchor_list'][i][1]]
        op_list.append(op)
        anchor_list.extend(op['anchor_list'])
    
    sampled_anchor_list = {}
    constraint_list = []
    for i in range(C):
        sample = random.sample(anchor_list, random.randint(4, 4))
        converted_sample = []
        for x in sample:
            x_name = convert_anchor_to_anchor_name(x[0])
            if x_name not in sampled_anchor_list:
                sampled_anchor_list[x_name] = x[0]
            converted_sample.append([x_name, x[1]])
        constraint = create_constraint(converted_sample)
        constraint_list.append(constraint)
    
    with open(filename, 'w') as f:
        for op in op_list:
            f.write('operation ' + op['name'] + ' ' + op['expr'] + '\n')
        f.write('\n')
        for anchor in sampled_anchor_list:
            if anchor != sampled_anchor_list[anchor]:
                f.write('anchor ' + anchor + " " + sampled_anchor_list[anchor] + '\n')
        f.write('\n')
        for constraint in constraint_list:
            f.write('constraint ' + constraint+'\n')
        f.write('\n')
        # write the timing for debugging
        for anchor in sampled_anchor_list:
            t = -1
            for a in anchor_list:
                if a[0] == sampled_anchor_list[anchor]:
                    t = a[1]
                    break
            if t == -1:
                logging.error('Cannot find anchor %s in the anchor list!' % anchor)
                sys.exit(1)
            f.write('# ' + anchor + ' = ' + str(t) + '\n')

def generate_testcases(N_VEC, M_VEC, C_VEC, directory):
    avg_N = N_VEC[int(len(N_VEC)/2)]
    avg_M = M_VEC[int(len(M_VEC)/2)]
    avg_C = C_VEC[int(len(C_VEC)/2)]

    for n in N_VEC:
        print(directory, 'N_%d' % n)
        curr_dir = os.path.join(directory, 'N_%d' % n)
        os.mkdir(curr_dir)
        m = avg_M
        c = avg_C
        for i in range(10):
            create_test_case(os.path.join(curr_dir, 'test_%d.txt' % (i)), n, m, c)
    
    for m in M_VEC:
        curr_dir = os.path.join(directory, 'M_%d' % m)
        os.mkdir(curr_dir)
        n = avg_N
        c = avg_C
        for i in range(10):
            create_test_case(os.path.join(curr_dir, 'test_%d.txt' % (i)), n, m, c)
    
    for c in C_VEC:
        curr_dir = os.path.join(directory, 'C_%d' % c)
        os.mkdir(curr_dir)
        n = avg_N
        m = avg_M
        for i in range(10):
            create_test_case(os.path.join(curr_dir, 'test_%d.txt' % (i)), n, m, c)

def run_test(N_VEC, M_VEC, C_VEC, directory):

    command = 'python module/vs-schedule/src/main.py -s schedule -m %s -o %s'

    testdir_list = []

    for n in N_VEC:
        curr_dir = os.path.join(directory, 'N_%d' % n)
        testdir_list.append(curr_dir)
    for m in M_VEC:
        curr_dir = os.path.join(directory, 'M_%d' % m)
        testdir_list.append(curr_dir)
    for c in C_VEC:
        curr_dir = os.path.join(directory, 'C_%d' % c)
        testdir_list.append(curr_dir)

    for testdir in testdir_list:
        curr_dir = testdir
        # check if the directory exists
        if not os.path.exists(curr_dir):
            logging.warning('Directory %s does not exist!' % curr_dir)
            sys.exit(1)
        time_records = []
        for i in range(10):
            filepath = os.path.join(curr_dir, 'test_%d.txt' % (i))
            if os.system(command % (filepath, curr_dir)) != 0:
                logging.error('Failed to run test %s' % filepath)
                sys.exit(1)
            # get execution time from output file: time.txt
            with open(os.path.join(curr_dir, 'time.txt'), 'r') as f:
                execution_time = float(f.readline())
            time_records.append(execution_time)
        avg_time = sum(time_records)/len(time_records)
        with open(os.path.join(curr_dir, 'time_record.txt'), 'w') as f:
            f.write(str(avg_time)+'\n')

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    directory = './testcases'
    if os.path.exists(directory) and os.path.isdir(directory) and os.listdir(directory) != []:
        logging.warning('Directory %s already exists and it is not empty!' % directory)
        # delete the folder
        os.system('rm -rf %s' % directory)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    N_VEC = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    M_VEC = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    C_VEC = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    testcase_dir =os.path.join(directory, 'testcases')
    if not os.path.exists(testcase_dir):
        os.mkdir(testcase_dir)
    generate_testcases(N_VEC, M_VEC, C_VEC, testcase_dir)

    run_test(N_VEC, M_VEC, C_VEC, testcase_dir)

    