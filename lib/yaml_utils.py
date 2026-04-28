import sympy as sp

# TYPES_DICTIONARY = {
#     "symbol": sp.Symbol
# }

# class IntList():
#     def __init__(self, vals):
#         if any([not isinstance(val, (int, sp.Integer)) for val in vals]):
#             raise TypeError(f"IntList must only contain int values. Given: {vals}")
#         self.list = [int(val) for val in vals]

#     def __eq__(self, other):
#         if not isinstance(other, IntList):
#             raise TypeError(f"IntList can only be compared to other IntList. Given: {other}")
#         return self.list == other.list
    
#     def __req__(self, other):
#         if not isinstance(other, IntList):
#             raise TypeError(f"IntList can only be compared to other IntList. Given: {other}")
#         return other.list == self.list
    
#     def __add__(self, other):
#         if not isinstance(other, IntList):
#             raise TypeError(f"IntList can only be added to other IntList. Given: {other}")
#         return other.list + self.list
    
#     def __radd__(self, other):
#         if not isinstance(other, IntList):
#             raise TypeError(f"IntList can only be added to other IntList. Given: {other}")
#         return self.list + other.list

def is_int_list(arg):
    return isinstance(arg, list) and all([isinstance(val, (int, sp.Integer)) for val in arg])

def evaluate_yaml_operators_tree(tree, variables):
    if isinstance(tree, (int, float)) or is_int_list(tree) or tree is None:
        return tree
    if isinstance(tree, str):
        if tree in variables:
            return variables[tree]
        elif tree == 'empty_list':
            return []
        else:
            raise ValueError(f"Variable {tree} not found in provided variables")
    # Handle operations
    if tree['op'] == 'add':
        return sum(evaluate_yaml_operators_tree(arg, variables) for arg in tree['args'])
    elif tree['op'] == 'mul':
        result = 1
        for arg in tree['args']:
            result *= evaluate_yaml_operators_tree(arg, variables)
        return result
    elif tree['op'] == 'sub':
        if len(tree['args']) != 2:
            raise ValueError("Subtraction operator requires exactly 2 arguments")
        return evaluate_yaml_operators_tree(tree['args'][0], variables) - evaluate_yaml_operators_tree(tree['args'][1], variables)
    elif tree['op'] == 'div':
        if len(tree['args']) != 2:
            raise ValueError("Division operator requires exactly 2 arguments")
        return evaluate_yaml_operators_tree(tree['args'][0], variables) / evaluate_yaml_operators_tree(tree['args'][1], variables)
    elif tree['op'] == 'eq':
        if len(tree['args']) != 2:
            raise ValueError("Equality operator requires exactly 2 arguments")
        return evaluate_yaml_operators_tree(tree['args'][0], variables) == evaluate_yaml_operators_tree(tree['args'][1], variables)
    elif tree['op'] == 'ne':
        if len(tree['args']) != 2:
            raise ValueError("Inequality operator requires exactly 2 arguments")
        return evaluate_yaml_operators_tree(tree['args'][0], variables) != evaluate_yaml_operators_tree(tree['args'][1], variables)
    elif tree['op'] == 'gt':
        if len(tree['args']) != 2:
            raise ValueError("Greater than operator requires exactly 2 arguments")
        return evaluate_yaml_operators_tree(tree['args'][0], variables) > evaluate_yaml_operators_tree(tree['args'][1], variables)
    elif tree['op'] == 'lt':
        if len(tree['args']) != 2:
            raise ValueError("Less than operator requires exactly 2 arguments")
        return evaluate_yaml_operators_tree(tree['args'][0], variables) < evaluate_yaml_operators_tree(tree['args'][1], variables)
    elif tree['op'] == 'ge':
        if len(tree['args']) != 2:
            raise ValueError("Greater than or equal operator requires exactly 2 arguments")
        return evaluate_yaml_operators_tree(tree['args'][0], variables) >= evaluate_yaml_operators_tree(tree['args'][1], variables)
    elif tree['op'] == 'le':
        if len(tree['args']) != 2:
            raise ValueError("Less than or equal operator requires exactly 2 arguments")
        return evaluate_yaml_operators_tree(tree['args'][0], variables) <= evaluate_yaml_operators_tree(tree['args'][1], variables)
    elif tree['op'] == 'slice':
        if len(tree['args']) != 3:
            raise ValueError("Slice operator requires exactly 3 arguments")
        orig_list = evaluate_yaml_operators_tree(tree['args'][0], variables)
        slice_idxs = slice(tree['args'][1], tree['args'][2])
        return orig_list[slice_idxs]
    elif tree['op'] == 'concat':
        if len(tree['args']) != 2:
            raise ValueError("Concat operator requires exactly 2 arguments")
        lists = [evaluate_yaml_operators_tree(arg, variables) for arg in tree['args']]
        if any([not is_int_list(arg) for arg in lists]):
            raise TypeError(f" Arguments of Concat operator must be lists of integers. Given: {tree['args']}")
        return lists[0] + lists[1]
    elif tree['op'] == 'and':
        return all(evaluate_yaml_operators_tree(arg, variables) for arg in tree['args'])
    elif tree['op'] == 'or':
        return any(evaluate_yaml_operators_tree(arg, variables) for arg in tree['args'])
    else:
        raise ValueError(f"Unsupported operator {tree['op']}")