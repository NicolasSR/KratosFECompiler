## Associated paper:
## https://www.sciencedirect.com/science/article/pii/S0045782523002001

from collections import OrderedDict

import numpy as np
import sympy as sp

from lib.basic_classes import DofsIndicator
from kratos_fe_compiler.compiler import compile

    
def main(options_dict):

    ## Symbolic generation settings
    dim = options_dict["dim"]

    # Define further options based on formulation
    if dim == 2:
        nnodes = 3
    elif dim == 3:
        nnodes = 4
    else:
        err_msg = "Incompatible number of dimensions: " + str(dim)
        raise Exception(err_msg)
    
    # Input for compiler

    use_transposed_gradients = False # If true, use convention dF_i/dx_j. Otherwise, use dF_j/dx_i
    trans_str = str(use_transposed_gradients)

    dofs = DofsIndicator('v','p')
    
    unknown_vars = [        # For each unknown var we need symbol, tensor rank and symbol for the corresponding test function
        {
            'symbol': 'v',
            'tensor_rank': 1,
            'test_function_symbol': 'w'
        },{
            'symbol': 'p',
            'tensor_rank': 0,
            'test_function_symbol': 'q'
        }]
    
    additional_nodal_tensors = [
        {
            'symbol': 'vn',
            'latex': 'v_n',
            'tensor_rank': 1
        },{
            'symbol': 'vnn',
            'latex': 'v_{nn}',
            'tensor_rank': 1
        },{
            'symbol': 'f',
            'tensor_rank': 1
        }]
    
    additional_function_tensors = []
    
    additional_symbolic_tensors = [
        {
            'symbol': 'vg',
            'latex': 'v_{g}',
            'tensor_rank': 1,
        },{
            'symbol': 'pg',
            'latex': 'p_{g}',
            'tensor_rank': 0,
        },{
            'symbol': 'grad_pg',
            'latex': '\nabla p_{g}',
            'tensor_rank': 1,
        },{
            'symbol': 'fg',
            'latex': 'f_{g}',
            'tensor_rank': 1,
        },{
            'symbol': 'acchg',
            'latex': '\dot{v}_{g}',
            'tensor_rank': 1,
        },{
            'symbol': 'stress',
            'latex': '\sigma',
            'tensor_rank': 2,
            'symmetric': True,
            'use_voigt_notation': True      # For symmetric tensors only
        },{
            'symbol': 'C',
            'tensor_rank': 4,
            'symmetric': True,          # For tensors of rank 2 and 4
            'third_symmetry': True,      # For tensors of rank 4 only
            'use_voigt_notation': True      # For symmetric tensors only
        },{
            'symbol': 'rho',
            'latex': '\rho',
            'tensor_rank': 0
        },{
            'symbol': 'tau1',
            'latex': '\tau_{1}',
            'tensor_rank': 0
        },{
            'symbol': 'tau2',
            'latex': '\tau_{2}',
            'tensor_rank': 0
        },{
            'symbol': 'bdf0',
            'latex': 'bdf_0',
            'tensor_rank': 0
        },{
            'symbol': 'bdf1',
            'latex': 'bdf_1',
            'tensor_rank': 0
        },{
            'symbol': 'bdf2',
            'latex': 'bdf_2',
            'tensor_rank': 0
        }
        ]
    
    numerical_tensors=[]
    
    defined_functions = [{
            'symbol': 'accel',
            'latex': '\dot{v}',
            'value': 'add_op(prod_op(¨bdf0¨,¨v¨),prod_op(¨bdf1¨,¨vn¨),prod_op(¨bdf2¨,¨vnn¨))'
        },{
            'symbol': 'vs',
            'latex': 'v_{s}',
            'value': f'prod_op(¨tau1¨,sub_op(¨fg¨,add_op(prod_op(¨rho¨,¨acchg¨),¨grad_pg¨)))'
        },{
            'symbol': 'ps',
            'latex': 'p_{s}',
            'value': '¨tau2¨*¨rho¨*div(¨v¨,¨base_scalars¨)'  # The original implementation seems to be missing a negative sign. Also for some reason it uses v instead of vg
        }
        ]

    expr = OrderedDict()

    expr['term1'] = "dot(¨w¨,¨fg¨)"
    expr['term2'] = "¨rho¨*dot(¨w¨,¨acchg¨)"
    expr['grad_of_w'] = f"grad(¨w¨,¨base_scalars¨,{trans_str})"
    expr['term3'] = "double_contract(¨grad_of_w¨,¨stress¨)"
    expr['term4'] = "div(¨w¨,¨base_scalars¨)*¨pg¨"
    expr['term5'] = "div(¨w¨,¨base_scalars¨)*¨ps¨"
    expr['term6'] = "-¨q¨*div(¨v¨,¨base_scalars¨)"   # The original implementation seems to be missing rho, so we did not include it here. Also for some reason it uses v instead og vg. 
    expr['term7'] = f"¨rho¨*dot(grad(¨q¨,¨base_scalars¨,{trans_str}),¨vs¨)"
    expr['vs_aux'] = 'prod_op(¨tau1¨,sub_op(¨f¨,prod_op(¨rho¨,¨accel¨)))'  # Because we ignore second derivatives, the vs we use in term8 will not include grad(p)
    expr['term8'] = "double_contract(¨grad_of_w¨,double_contract(¨C¨,symgrad(¨vs_aux¨,¨base_scalars¨)))"

    expr['functional'] = "¨term1¨ - ¨term2¨ - ¨term3¨ + ¨term4¨ + ¨term5¨ + ¨term6¨ + ¨term7¨ - ¨term8¨"

    # Explicit stress to substitute later
    expr['expl_stress'] = "double_contract(¨C¨,symgrad(¨v¨,¨base_scalars¨))"
    expr['expl_grad_p'] = f"grad(¨p¨,¨base_scalars¨,{trans_str})"

    functional_rhs_name = 'functional'
    functional_lhs_name = 'functional'

    pre_lhs_substitutions = [
        ('fg', 'f'),
        ('vg', 'v'),
        ('grad_pg', 'expl_grad_p'),
        ('pg', 'p'),
        ('acchg', 'accel'),
        ('stress', 'expl_stress')
    ]
    post_lhs_substitutions = []

    return compile(dim, nnodes, use_transposed_gradients, unknown_vars, additional_nodal_tensors, additional_function_tensors, additional_symbolic_tensors,
              numerical_tensors, defined_functions, expr, functional_rhs_name, functional_lhs_name, pre_lhs_substitutions, post_lhs_substitutions, dofs)

