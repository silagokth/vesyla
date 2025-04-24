import re


def add_constraint(cstr_type, expr, constraint_list):
    expr = re.sub(r"\s+", "", expr)
    constraint_list.append([cstr_type, expr])
