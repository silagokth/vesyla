
import regex as re
def add_constraint(expr, constraint_list):
    expr = re.sub(r"\s+", "", expr)
    constraint_list.append(expr)