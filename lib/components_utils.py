import itertools

import sympy as sp

def get_flat_list_of_components(var):
    iterate = False
    if isinstance(var, sp.tensor.array.ndim_array.NDimArray):
        rank = var.rank()
        dim = var.shape[0]
        index_ranges = [range(dim) for _ in range(rank)]
        iterate = True
    elif isinstance(var, sp.MatrixBase):
        index_ranges = [range(dim) for dim in var.shape]
        iterate = True
    if iterate:
        index_combinations = itertools.product(*index_ranges)
        return [var[indices] for indices in index_combinations]
    else:
        return [var]

# def get_tuple_subelements(tuple_arg):
#     from lib.basic_classes import DerivIndicator
#     for arg in tuple_arg:
#         out = []
#         if isinstance(arg, DerivIndicator):
#             out.extend(arg.get_args_names_list())
#         elif isinstance(arg, str):
#             out.append(arg)
#         else:
#             raise "Substitution tuples should contain only strings or DerivIndicators"
#     return out