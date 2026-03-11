import sympy as sp


def DefineVector( name, m):
    """
    This method defines a symbolic vector.

    Keyword arguments:
    - name -- Name of variables.
    - m -- Number of components.
    """
    return sp.Matrix(m, 1, lambda i,_: sp.var("{name}_{i}".format(name=name, i=i)))

def DefineMatrix(name, m, n):
    """
    This method defines a symbolic matrix.

    Keyword arguments:
    - name -- Name of variables.
    - m -- Number of rows.
    - n -- Number of columns.
    """
    return sp.Matrix(m, n, lambda i, j: sp.var("{name}_{i}_{j}".format(name=name, i=i, j=j)))

def DefineSymmetricMatrix(name, m, n):
    """
    This method defines a symbolic symmetric matrix.

    Keyword arguments:
    - name -- Name of variables.
    - m -- Number of rows.
    - n -- Number of columns.
    """
    return sp.Matrix(m, n, lambda i, j:
        sp.var("{name}_{i}_{j}".format(name=name, i=min(i,j), j=max(i,j))))

def DefineZeroOrderTensorField(name, base_scalars, is_positive=False):
    """
    This method defines a symbolic 0th order tensor field.
    Just a scalar function depending on the variables in base_scalars

    Keyword arguments:
    - name -- Name of variables.
    - base_scalars -- list of variables on which the field depends.
    """
    # return ZeroOrderTensor(sp.Function(name)(*base_scalars))
    if base_scalars is not None:
        return sp.Function(name)(*base_scalars)
    else:
        return sp.Symbol(name,positive=is_positive)

def DefineFirstOrderTensorField(name, dim, base_scalars):
    """
    This method defines a symbolic 1st order tensor field.
    All the elements are fundtions of the variables in base_scalars

    Keyword arguments:
    - name -- Name of variables.
    - dim -- dimension of the tensor.
    - base_scalars -- list of variables on which the field depends.
    """
    if base_scalars is not None:
        return sp.Array([sp.Function(name+'_'+str(i))(*base_scalars) for i in range(dim)])
    else:
        return sp.Array([sp.Symbol(name+'_'+str(i)) for i in range(dim)])

def DefineSecondOrderTensorField(name, m, n, base_scalars):
    """
    This method defines a symbolic 2nd order tensor field.
    All the elements are fundtions of the variables in base_scalars

    Keyword arguments:
    - name -- Name of variables.
    - m -- first dimension of the tensor.
    - n -- second dimension of the tensor.
    - base_scalars -- list of variables on which the field depends.
    """
    if m != n:
        raise ValueError("Provided sizes do not respect symmetry.")
    tensor = sp.MutableDenseNDimArray(sp.zeros(m**2),shape=(m,n))
    for i in range(m):
        for j in range(n):
            unique_index = [i, j] # Apply symmetry
            indexes_string=''
            for index in unique_index:
                indexes_string += '_'+str(index)
            if base_scalars is not None:
                tensor[i,j] = sp.Function(name+indexes_string)(*base_scalars)
            else:
                tensor[i,j] = sp.Symbol(name+indexes_string)
    return tensor

def DefineSymmetricSecondOrderTensorField(name, m, n, base_scalars):
    """
    This method defines a symbolic symmetric 2nd order tensor field.
    All the elements are fundtions of the variables in base_scalars

    Keyword arguments:
    - name -- Name of variables.
    - m -- first dimension of the tensor.
    - n -- second dimension of the tensor.
    - base_scalars -- list of variables on which the field depends.
    """
    if m != n:
        raise ValueError("Provided sizes do not respect symmetry.")
    tensor = sp.MutableDenseNDimArray(sp.zeros(m**2),shape=(m,n))
    for i in range(m):
        for j in range(n):
            unique_index = [min(i,j), max(i,j)] # Apply symmetry
            indexes_string=''
            for index in unique_index:
                indexes_string += '_'+str(index)
            if base_scalars is not None:
                tensor[i,j] = sp.Function(name+indexes_string)(*base_scalars)
            else:

                tensor[i,j] = sp.Symbol(name+indexes_string)
    return tensor
    
def DefineSymmetricFourthOrderTensorField(name, m, n, o, p, base_scalars, apply_third_symmetry = True):
    """
    This method defines a symbolic symmetric 4th order tensor field.
    Symmetries are the following: Aijkl=Ajikl, Aijkl=Aijlk.
    If applying third symmetry, also: Aijkl=Aklij
    All the elements are fundtions of the variables in base_scalars

    Keyword arguments:
    - name -- Name of variables.
    - m -- 1st dimension.
    - n -- 2nd dimension.
    - o -- 3rd dimension.
    - p -- 4th dimension.
    - base_scalars -- list of variables on which the field depends.
    - apply_third_symmetry -- whether to apply the third symmetry
    """

    if m != n:
        raise ValueError("Provided sizes do not respect first symmetry.")
    if o != p:
        raise ValueError("Provided sizes do not respect second symmetry.")
    if apply_third_symmetry and m!=o:
        raise ValueError("Provided sizes do not respect third symmetry.")

    tensor = sp.MutableDenseNDimArray(sp.zeros(m**4),shape=(m,n,o,p))
    for i in range(m):
        for j in range(n):
            for k in range(o):
                for l in range(p):
                    unique_index = [min(i,j), max(i,j), min(k,l), max(k,l)] # Apply first two symmetries
                    if apply_third_symmetry:
                        if unique_index[0] > unique_index[2]:
                            unique_index = [unique_index[2],unique_index[3],unique_index[0],unique_index[1]]
                        elif unique_index[0] == unique_index[2]:
                            if unique_index[1] > unique_index[3]:
                                unique_index = [unique_index[2],unique_index[3],unique_index[0],unique_index[1]]
                    indexes_string=''
                    for index in unique_index:
                        indexes_string += '_'+str(index)
                    if base_scalars is not None:
                        tensor[i,j,k,l] = sp.Function(name+indexes_string)(*base_scalars)
                    else:
                        tensor[i,j,k,l] = sp.Symbol(name+indexes_string)
    return tensor

# def DefineSymmetricFourthOrderTensor(name, m, n, o, p):
#     """
#     This method defines a symbolic symmetric 4th order tensor.

#     Keyword arguments:
#     - name -- Name of variables.
#     - m -- 1st dimension.
#     - n -- 2nd dimension.
#     - o -- 3rd dimension.
#     - p -- 4th dimension.
#     """
#     if m != n:
#         raise ValueError("Provided sizes do not respect first symmetry.")
#     if o != p:
#         raise ValueError("Provided sizes do not respect second symmetry.")

#     tensor = sp.MutableDenseNDimArray(sp.zeros(m**4),shape=(m,n,o,p))
#     for i in range(m):
#         for j in range(n):
#             for k in range(o):
#                 for l in range(p):
#                     tensor[i,j,k,l] = sp.var("{name}_{m}_{n}_{o}_{p}".format(name=name, m=min(i,j), n=max(i,j), o=min(k,l), p=max(k,l)))

#     return tensor
