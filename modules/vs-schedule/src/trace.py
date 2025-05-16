import json
from binarytree import Node
import operation
import itertools


class TraceEvent:
    def __init__(self, name, cat, ph, ts, pid, tid):
        self.name = name
        self.cat = cat
        self.ph = ph
        self.ts = ts
        self.pid = pid
        self.tid = tid

    def to_json(self):
        return json.dumps(self.__dict__)


class OperationEvent:
    def __init__(self, op_name, event_id, time):
        self.op_name = op_name
        self.event_id = event_id
        self.time = time


def convert_operation_event_to_trace_event_list(op_event, counter, op2pid_map):
    if op_event.op_name not in op2pid_map:
        op2pid_map[op_event.op_name] = counter[0]
        counter[0] += 1
    pid = op2pid_map[op_event.op_name]
    tid = op_event.event_id
    name = op_event.op_name + ".e" + str(op_event.event_id)
    cat = op_event.op_name
    return [
        TraceEvent(name, cat, "B", op_event.time, pid, tid),
        TraceEvent(name, cat, "E", op_event.time + 1, pid, tid),
    ]


def create_event_list():
    event_list = []
    event_list.append(OperationEvent("op1", 1, 0))
    event_list.append(OperationEvent("op1", 2, 1))
    event_list.append(OperationEvent("op2", 1, 2))
    event_list.append(OperationEvent("op2", 2, 3))
    event_list.append(OperationEvent("op1", 3, 4))
    return event_list


def create_trace_file_from_event_list(event_list, filename):
    op_event_list = event_list
    trace_event_list = []
    op2pid_map = {}
    counter = [0]
    for op_event in op_event_list:
        converted_list = convert_operation_event_to_trace_event_list(
            op_event, counter, op2pid_map
        )
        trace_event_list.extend(converted_list)
    with open(filename, "w") as f:
        f.write("[\n")
        for trace_event in trace_event_list:
            if trace_event != trace_event_list[-1]:
                f.write(trace_event.to_json() + ",\n")
            else:
                f.write(trace_event.to_json() + "\n")
        f.write("]\n")


def create_event_list_from_btree(
    node: Node, op_name, time, node_map, start_time_table, duration_table
):
    event_list = []
    if node is None:
        return event_list

    # perform a traversal, when a leaf node is reached, its full parents path is stored in parents list
    event_counter = 0
    parent_list = []
    current = node
    while current is not None:
        while current.left is not None:
            parent_list.append(current)
            current = current.left

        # a leaf node is reached
        if isinstance(node_map[current.value], operation.Event):
            event_start_time = time + int(start_time_table[current.value])
            # It is an Event node
            # get all the parents of the current node, including only the RepeatitionOperator node
            parent_operator_node_list = [
                x
                for x in parent_list
                if isinstance(node_map[x.value], operation.RepeatitionOperator)
            ]

            operator_info_list = []
            for node in parent_operator_node_list:
                id = node.value
                D = duration_table[node.left.value]
                tau = node_map[id].delay_var
                iter = node_map[id].iter
                operation_info = {"D": D, "tau": tau, "iter": iter}
                operator_info_list.append(operation_info)

            iter_list = [int(x["iter"]) for x in operator_info_list]
            all_indices = [
                list(x) for x in itertools.product(*[range(i) for i in iter_list])
            ]

            for indices in all_indices:
                event_time = event_start_time
                for i in range(len(parent_operator_node_list)):
                    id = parent_operator_node_list[i].value
                    D = operator_info_list[i]["D"]
                    tau = operator_info_list[i]["tau"]
                    iter = operator_info_list[i]["iter"]
                    event_time += (int(D) + int(tau)) * indices[i]
                event_list.append(OperationEvent(op_name, event_counter, event_time))

            event_counter += 1

        # go to a level above and continue until all leaf nodes are reached
        while parent_list and (
            not parent_list[-1].right or parent_list[-1].right == current
        ):
            current = parent_list.pop()

        if parent_list:
            current = parent_list[-1].right
        else:
            current = None

    return event_list
