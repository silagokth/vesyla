from ortools.sat.python import cp_model
import logging

def solve(timing_var_dict, constraint_list, op_table, duration_table, max_latency: int):
    cp_vars = {}
    model = cp_model.CpModel()
    for name in timing_var_dict:
        if name == timing_var_dict[name]:
            cp_vars[name] = model.NewIntVar(0, max_latency, name)
    for name in timing_var_dict:
        if name != timing_var_dict[name]:
            cp_vars[name] = model.NewIntVar(0, max_latency, name)
            model.Add(cp_vars[name] == (eval(timing_var_dict[name], cp_vars)))
    
    for cstr in constraint_list:
        if cstr[0] == "linear":
            model.Add(eval(cstr[1], cp_vars))
        elif cstr[0] == "all_different":
            model.AddAllDifferent(eval(cstr[1], cp_vars))
    
    op_end_time_list = []
    for op_name in op_table:
        expr = "(" + timing_var_dict[op_name] + ")+(" + duration_table[op_name][op_table[op_name].value] + ")"
        var = model.NewIntVar(0, max_latency, op_name+"_end_time")
        model.Add(var == (eval(expr, cp_vars)))
        op_end_time_list.append(var)
    
    obj = model.NewIntVar(0, max_latency, "obj")
    model.AddMaxEquality(obj, op_end_time_list)

    # minimize obj
    model.Minimize(obj)


    # solve the model
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    if status == cp_model.OPTIMAL:
        solution = {}
        for name in timing_var_dict:
            solution[name] = solver.Value(cp_vars[name])
        latency = solver.Value(obj)
        return [latency, solution]
    else:
        logging.error("No solution found")
        exit(1)
        return None