## Associated paper:
## https://www.sciencedirect.com/science/article/pii/S0045782523002001

from collections import OrderedDict

import numpy as np

# from sympy_utilities.my_sympy_utilities import grad, det, cofactor, double_contract, matrix_prod, transpose, dot, matrix_vector_prod
from lib.basic_classes import DofsIndicator, DerivIndicator
from kratos_fe_compiler.compiler import compile

def main(options_dict):

    ## User inputs:
    dim = 2
    nnodes = 3

    use_transposed_gradients = True # If true, use convention dF_i/dx_j. Otherwise, use dF_j/dx_i
    trans_str = str(use_transposed_gradients)

    unknown_vars = [        # For each unknown var we need symbol, tensor rank and symbol for the corresponding test function
        {
            'symbol': 'u',
            'tensor_rank': 1,
            'test_function_symbol': 'w'
        },{
            'symbol': 'theta',
            'latex': '\\theta',
            'tensor_rank': 0,
            'test_function_symbol': 'q'
        }]
    dofs = DofsIndicator('u','theta')
    
    additional_nodal_tensors = []
    
    additional_function_tensors = [
        {
            'symbol': 'Efun',
            'tensor_rank': 2,
            'dependencies': dofs
        },{
            'symbol': 'Sfun',
            'tensor_rank': 2,
            'dependencies': 'Efun'
        }]
    
    additional_symbolic_tensors = [
        {
            'symbol': 'S',
            'tensor_rank': 2,
            'symmetric': True,          # For tensors of rank 2 and 4
            'use_voigt_notation': True      # For symmetric tensors only
        },{
            'symbol': 'C',
            'tensor_rank': 4,
            'symmetric': True,          # For tensors of rank 2 and 4
            'third_symmetry': True,      # For tensors of rank 4 only
            'use_voigt_notation': True      # For symmetric tensors only
        },{
            'symbol': 'b_o',
            'latex': 'b_{o}',
            'tensor_rank': 1
        },{
            'symbol': 'rho_o',
            'latex': '\\rho_{o}',
            'tensor_rank': 0
        },{
            'symbol': 'tau_theta',
            'latex': '\\tau_{\\theta}',
            'tensor_rank': 0,
            'positive': True
        },{
            'symbol': 'tau_u',
            'latex': '\\tau_{u}',
            'tensor_rank': 0,
            'positive': True
        }]
    
    numerical_tensors = [
        {
            'symbol': 'I',
            'tensor_rank': 2,
            'value': np.eye(dim, dtype=int)
        },{
            'symbol': 'd',
            'tensor_rank': 0,
            'value': dim
        }]
    
    defined_functions = [
        {
            'symbol': 'F',
            'value': f"add_op(¨I¨,grad(¨u¨,¨base_scalars¨,{trans_str}))"
        },{
            'symbol': 'J',
            'value': "det(¨F¨)"
        },{
            'symbol': 'cofF',
            'latex': '\\mathrm{cof}F',
            'value': "cofactor(¨F¨)"
        }
    ]
    

    expr = OrderedDict()
    expr['term1']       = "dot(¨w¨,prod_op(¨rho_o¨,¨b_o¨))"
    expr['term2']       = f"double_contract(grad(¨w¨,¨base_scalars¨,{trans_str}),matrix_prod(¨F¨,¨S¨))"
    expr['term3_aux1']  = "¨theta¨**(sub_op(2,¨d¨)/¨d¨)/¨d¨*¨J¨**(-2/¨d¨)*¨tau_theta¨*(¨theta¨-¨J¨)"
    expr['term3_aux2']  = "double_contract(¨C¨,matrix_prod(transpose(¨F¨),¨F¨))"
    expr['term3']       = f"¨term3_aux1¨*double_contract(grad(¨w¨,¨base_scalars¨,{trans_str}),¨term3_aux2¨)"
    expr['term4']       = "(1-¨tau_theta¨)*¨q¨*(¨theta¨-¨J¨)"
    expr['term5_aux']   = f"matrix_vector_prod(¨cofF¨,grad(¨q¨,¨base_scalars¨,{trans_str}))"
    expr['term5']       = "dot(¨term5_aux¨,prod_op(¨tau_u¨*¨rho_o¨,¨b_o¨))"
    expr['term6_aux']   = "¨tau_u¨/¨d¨*(¨J¨/¨theta¨)**(sub_op(¨d¨,2)/¨d¨)"
    expr['term6']       = f"¨term6_aux¨*dot(grad(¨q¨,¨base_scalars¨,{trans_str}),matrix_vector_prod(¨term3_aux2¨,grad(¨theta¨,¨base_scalars¨,{trans_str})))"

    expr['functional']  = "¨term1¨ - ¨term2¨ + ¨term3¨ + ¨term4¨ + ¨term5¨ + ¨term6¨"
    # expr['functional']  = "¨term5¨"

    expr['F_bar']       = "prod_op(¨J¨**(-1/¨d¨),¨F¨)"
    expr['F_bar_sq']    = "matrix_prod(transpose(¨F_bar¨),¨F_bar¨)"
    expr['E']           = "prod_op(0.5,sub_op(prod_op(¨theta¨**(2/¨d¨),¨F_bar_sq¨),¨I¨))"

    functional_rhs_name = functional_lhs_name = 'functional'

    # Substitutions before LHS
    pre_lhs_substitutions = [
        ('S',       'Sfun')
        ]

    # Substitutions after LHS
    post_lhs_substitutions = [
        (DerivIndicator('Sfun','Efun'),     'C'),
        (DerivIndicator('Efun',dofs),       DerivIndicator('E',dofs)),
        ('Sfun',    'S'),
        ('Efun',    'E')
        ]

    return compile(dim, nnodes, use_transposed_gradients, unknown_vars, additional_nodal_tensors, additional_function_tensors, additional_symbolic_tensors,
              numerical_tensors, defined_functions, expr, functional_rhs_name, functional_lhs_name, pre_lhs_substitutions, post_lhs_substitutions, dofs)
