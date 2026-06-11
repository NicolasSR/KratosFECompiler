from typing import Tuple

from sympy import NDimArray

import sympy as sp

def tensor_grad(var:sp.Array, base_scalars: Tuple):
    return sp.derive_by_array(var, sp.Array([*base_scalars]))

def tensor_mat_transpose(a:sp.Array):
    assert a.rank()==2
    return sp.permutedims(a,(1,0))

def tensor_grad_transposed(var:sp.Array, base_scalars: Tuple):
    grad = sp.derive_by_array(var, base_scalars)
    var_rank = sp.Array(var).rank()
    base_scalars_rank = sp.Array([*base_scalars]).rank()
    permutation = [i for i in range(base_scalars_rank,base_scalars_rank+var_rank)]
    permutation.extend([i for i in range(base_scalars_rank)])
    return sp.permutedims(grad,permutation)

def tensor_symgrad(var:sp.Array, base_scalars: Tuple):
    grad = sp.derive_by_array(var, base_scalars)
    grad_t = tensor_mat_transpose(grad.copy())
    return(sp.Rational(1,2)*(grad+grad_t))

def tensor_div(var:sp.Array, base_scalars: Tuple):
    var_grad = tensor_grad(var, base_scalars)
    return sp.tensorcontraction(var_grad, (0, 1))

def _get_levi_civita_array(dim):
    e = []
    for i in range(dim):
        e.append([])
        for j in range(dim):
            if dim==2:
                e[-1].append(sp.LeviCivita(i, j))
            elif dim==3:
                e[-1].append([])
                for k in range(dim):
                    e[-1][-1].append(sp.LeviCivita(i, j, k))
            else:
                raise Exception("Only dimensions 2 or 3 accepted for Levi-Civita array generation")
    return sp.Array(e)

def tensor_3D_curl(var:sp.Array, base_scalars: Tuple):
    # curl = e_ijk * D_j(u_k)
    # e_ijk = 3D Levi-Civita symbol
    assert var.rank()==1 & var.shape[0]==3
    grad = tensor_grad(var, base_scalars)
    e_tensor = _get_levi_civita_array(3)
    aux = sp.tensorproduct(e_tensor,grad)
    return sp.tensorcontraction(sp.tensorcontraction(aux, (1,3)), (1,2))

def tensor_2D_curl(var:sp.Array, base_scalars: Tuple):
    # curl = e_jk * D_j(u_k)
    # e_jk = 2D Levi-Civita symbol
    assert var.rank()==1 & var.shape[0]==2
    grad = tensor_grad(var, base_scalars)
    e_tensor = _get_levi_civita_array(2)
    aux = sp.tensorproduct(e_tensor,grad)
    return sp.tensorcontraction(sp.tensorcontraction(aux, (0,2)), (0,1))

def tensor_vector_dot(a: sp.Array, b: sp.Array):
    assert a.rank()==1 and b.rank()==1
    return sp.tensorcontraction(sp.tensorproduct(a,b),(0,1))

def tensor_vector_norm(a: sp.Array):
    assert a.rank()==1
    return sp.sqrt(tensor_vector_dot(a,a))

def tensor_singlecontract(a: sp.Array, b: sp.Array):
    assert a.rank()>=1 and b.rank()>=1
    return sp.tensorcontraction(sp.tensorproduct(a,b),(a.rank()-1,a.rank()))

# def tensor_doublecontract(a: sp.Array, b: sp.Array):
#     assert a.rank()>=2 and b.rank()>=2
#     # a_ijkl:b_mn -> c_ijklmn -> c_ijklml = d_ijkm-> d_ijkk = c_ijklkl
#     # a_ijk:b_lmn -> c_ijklmn -> c_ijklkn = d_ijln-> d_ijjn = c_ijkjkn
#     # a_ij:b_lm -> c_ijlm -> c_ijlj = d_il-> d_ii = c_ijij = a_ij:b:ij
#     # For tensors of rank 2 it's the same as UFL's inner product. For higher ranks it differs.
#     contracted = sp.tensorcontraction(sp.tensorproduct(a,b),(a.rank()-1,a.rank()+1))
#     return sp.tensorcontraction(contracted, (a.rank()-2,a.rank()-1))

def tensor_doublecontract(a: sp.Array, b: sp.Array):
    assert a.rank()>=2 and b.rank()>=2
    # a_ijk:b_lmn -> c_ijklmn -> c_ijkjmn -> d_in = c_ijkjkn = a_ijk b_jkn
    # a_ij:b_lm -> c_ijlm -> c_ijim = d_jm-> d_jj = c_ijij = a_ij b_ij
    # For tensors of rank 2 it's teh same as UFL's inner product. For higher ranks it differs.
    contracted = sp.tensorcontraction(sp.tensorproduct(a,b),(a.rank()-2,a.rank()))
    return sp.tensorcontraction(contracted, (a.rank()-2,a.rank()-1))

# def tensor_inner(a: sp.Array, b: sp.Array):
#     assert a.rank()>=2 and b.rank()>=2
#     contracted = sp.tensorcontraction(sp.tensorproduct(a,b),(a.rank()-2,a.rank()))
#     return sp.tensorcontraction(contracted, (a.rank()-2,a.rank()-1))

def tensor_sym_2D(a: sp.Array):
    assert a.rank()==2
    return (a+sp.permutedims(a,(1,0)))/2

def tensor_mat_prod(a:sp.Array, b:sp.Array):
    assert a.rank()==2 and b.rank()==2
    return sp.tensorcontraction(sp.tensorproduct(a,b),(1,2))

def tensor_mat_vector_prod(a:sp.Array, b:sp.Array):
    assert a.rank()==2 and b.rank()==1
    return sp.tensorcontraction(sp.tensorproduct(a,b),(1,2))

def tensor_mat_det(a:sp.Array):
    assert a.rank()==2 and a.shape[0]==a.shape[1]
    dim = a.shape[0]

    e_tensor = _get_levi_civita_array(dim)

    if dim == 2:
        det_aux = sp.tensorproduct(e_tensor,a[0],a[1])
        det = sp.tensorcontraction(sp.tensorcontraction(det_aux, (0,2)),(0,1))
    elif dim == 3:
        det_aux = sp.tensorproduct(e_tensor,a[0],a[1],a[2])
        det = sp.tensorcontraction(sp.tensorcontraction(sp.tensorcontraction(det_aux, (0,3)),(0,2)),(0,1))
    
    return det

def tensor_mat_cofactor(a:sp.Array):
    assert a.rank()==2 and a.shape[0]==a.shape[1]
    dim = a.shape[0]
   
    e_tensor = _get_levi_civita_array(dim)

    if dim == 2:
        cof_aux = sp.tensorproduct(e_tensor,e_tensor,a)
        cof = sp.tensorcontraction(sp.tensorcontraction(cof_aux, (1,4)),(2,3))
    elif dim == 3:
        cof_aux = sp.tensorproduct(e_tensor,e_tensor,a,a)
        cof = sp.Rational(1,2)*sp.tensorcontraction(sp.tensorcontraction(sp.tensorcontraction(sp.tensorcontraction(cof_aux, (1,6)),(1,6)),(2,4)),(2,3))
    
    return cof

def tensor_mat_inv(a:sp.Array):
    assert a.rank()==2 and a.shape[0]==a.shape[1]
    return tensor_mat_transpose(tensor_mat_cofactor(a))*tensor_mat_det(a)**(-1)

def tensor_vec_cross_prod(a:sp.Array,b:sp.Array):
    assert a.rank()==1 and b.rank()==1 and a.shape[0]==b.shape[0]
    dim = a.shape[0]
    e_tensor = _get_levi_civita_array(dim)
    if dim == 3:
        cross_aux = sp.tensorproduct(a,b,e_tensor)
        cross = sp.tensorcontraction(sp.tensorcontraction(cross_aux, (0,2)),(0,1))
    else:
        raise Exception("Only 3D vectors accepted for cross product computation")
    return cross

def tensor_vec_outer_prod(a:sp.Array,b:sp.Array):
    assert a.rank()==1 and b.rank()==1
    return sp.tensorproduct(a,b)

def tensor_arrays_flatten_and_combine(input_arrays: Tuple):
    flat_inputs_list = []
    for array in input_arrays:
        if isinstance(array, NDimArray):
            array_list = array.tolist()
            for comp in range(array.rank()-1):
                array_list_aux=[]
                for el in array_list:
                    array_list_aux.extend(el)
                array_list = array_list_aux
            flat_inputs_list.extend(array_list)
        else:
            flat_inputs_list.append(array)
    return sp.Array(flat_inputs_list)