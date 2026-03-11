import sympy as sp

from lib.array_definitions import DefineVector, DefineMatrix

def DefineShapeFunctions(nnodes, dim, impose_partion_of_unity=False, shape_functions_name='N', first_derivatives_name='DN', second_derivatives_name=None):
    """
    This method defines shape functions and derivatives.
    Second order derivatives can be optionally defined as well.
    Note that partition of unity can be imposed.

    Keyword arguments:
    - nnodes -- Number of nodes
    - dim -- Dimension of the space
    - impose_partion_of_unity -- Impose the partition of unity
    - shape_functions_name -- Name for the shape functions symbols
    - first_derivatives_name -- Name for the shape functions first derivatives symbols
    - second_derivatives_name -- Name for the shape functions second derivatives symbols
    """
    DN = DefineMatrix(first_derivatives_name, nnodes, dim)
    N = DefineVector(shape_functions_name, nnodes)

    # Impose partition of unity
    if impose_partion_of_unity:
        N[nnodes-1] = 1
        for i in range(nnodes-1):
            N[nnodes-1] -= N[i]

        DN[nnodes-1,:] = -DN[0,:]
        for i in range(1,nnodes-1):
            DN[nnodes-1,:] -= DN[i,:]

    if second_derivatives_name:
        if not impose_partion_of_unity:
            DDN = sp.Matrix(1, nnodes, lambda _, j : (sp.Matrix(dim, dim, lambda m, n : sp.var(f"{second_derivatives_name}_{j}_{m}_{n}"))))
        else:
            raise Exception("Partition of unity imposition is not implemented for shape functions second derivatives.")

    if second_derivatives_name:
        return N, DN, DDN
    else:
        return N, DN

def DfjDxi(DN,f):
    """
    This method defines a gradient. Returns a matrix D such that D(i,j) = D(fj)/D(xi)

    This is the standard in fluid dynamics, that is:
        D(f1)/D(x1) D(f2)/D(x1) D(f3)/D(x1)
        D(f1)/D(x2) D(f2)/D(x2) D(f3)/D(x2)
        D(f1)/D(x3) D(f2)/D(x3) D(f3)/D(x3)

    Keyword arguments:
    - DN -- The shape function derivatives
    - f-- The variable to compute the gradient
    """
    return sp.simplify(DN.transpose()*f)

def DfiDxj(DN,f):
    """
    This method defines a gradient This returns a matrix D such that D(i,j) = D(fi)/D(xj).

    This is the standard in structural mechanics, that is:
        D(f1)/D(x1) D(f1)/D(x2) D(f1)/D(x3)
        D(f2)/D(x1) D(f2)/D(x2) D(f2)/D(x3)
        D(f3)/D(x1) D(f3)/D(x2) D(f3)/D(x3)

    Keyword arguments:
    - DN -- The shape function derivatives
    - f -- The variable to compute the gradient
    """
    return (DfjDxi(DN,f)).transpose()

def div(DN,x):
    """
    This method defines the divergence.

    Keyword arguments:
    - DN -- The shape function derivatives
    - x -- The variable to compute the gradient
    """
    if DN.shape != x.shape:
        raise ValueError("shapes are not compatible")

    div_x = 0
    for i in range(DN.shape[0]):
        for k in range(DN.shape[1]):
            div_x += DN[i,k]*x[i,k]

    return sp.Matrix([sp.simplify(div_x)])

def Compute_RHS(functional, testfunc, do_simplifications=False):
    """
    This computes the RHS vector.

    Keyword arguments:
    - functional -- The functional to derivate
    - testfunc -- The test functions
    - do_simplifications -- If apply simplifications
    """
    rhs = sp.Matrix(sp.zeros(testfunc.shape[0],1))
    for i in range(testfunc.shape[0]):
        rhs[i] = sp.diff(functional, testfunc[i])

        if do_simplifications:
            rhs[i] = sp.simplify(rhs[i])

    return rhs

def Compute_LHS(rhs, testfunc, dofs, do_simplifications=False):
    """
    This computes the LHS matrix.

    Keyword arguments:
    - rhs -- The RHS vector
    - testfunc -- The test functions
    - dofs -- The dofs vectors
    - do_simplifications -- If apply simplifications
    """
    lhs = sp.Matrix(sp.zeros(testfunc.shape[0],dofs.shape[0]) )
    for i in range(lhs.shape[0]):
        for j in range(lhs.shape[1]):
            lhs[i,j] = -sp.diff(rhs[i,0], dofs[j,0])

            if do_simplifications:
                lhs[i,j] = sp.simplify(lhs[i,j])

    return lhs