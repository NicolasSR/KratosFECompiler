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