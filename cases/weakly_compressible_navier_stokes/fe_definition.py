## Associated paper:
## https://www.sciencedirect.com/science/article/pii/S0045782523002001

from collections import OrderedDict

import numpy as np
import sympy as sp

from lib.basic_classes import DofsIndicator
from kratos_fe_compiler.compiler import compile

INPUT_COMPATIBILITIES_DICT = {
    "WeaklyCompressibleNavierStokes": {
        2: [3],
        3: [4]
    },
    "Stokes": {
        2: [3,4],
        3: [4,6,8]
    }
}
    
def main(options_dict):

    ## Symbolic generation settings
    dim = options_dict["dim"]
    nnodes = options_dict["nnodes"]
    formulation = options_dict["formulation"]
    divide_by_rho = options_dict["divide_by_rho"]
    ASGS_stabilization = options_dict["ASGS_stabilization"]

    # Define further options based on formulation
    if formulation == "WeaklyCompressibleNavierStokes":
        darcy_term = True
        convective_term = True
        artificial_compressibility = True
        linearisation = "FullNR" # Convective term linearisation type. Options: "Picard", "FullNR"
    elif formulation == "Stokes":
        darcy_term = False
        convective_term = False
        artificial_compressibility = False
    else:
        err_msg = "Wrong formulation. Given \'" + formulation + "\'. Available options are \'WeaklyCompressibleNavierStokes\' and \'Stokes\'."
        raise Exception(err_msg)
    
    ## Check that formulation type, dim and number of nodes are compatible:
    if (not dim in INPUT_COMPATIBILITIES_DICT[formulation].keys()) or (not nnodes in INPUT_COMPATIBILITIES_DICT[formulation][dim]):
        err_msg = "Wrong Dimensions or Number of Nodes for formulation " + formulation
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
        },{
            'symbol': 'v_sol_frac',
            'latex': 'v_{SolFrac}',
            'tensor_rank': 1
        }
        ]
    
    additional_function_tensors = []
    
    additional_symbolic_tensors = [
        {
            'symbol': 'C',
            'tensor_rank': 4,
            'symmetric': True,          # For tensors of rank 2 and 4
            'third_symmetry': True,      # For tensors of rank 4 only
            'use_voigt_notation': True      # For symmetric tensors only
        },{
            'symbol': 'stress',
            'tensor_rank': 2,
            'symmetric': True,          # For tensors of rank 2 and 4
            'use_voigt_notation': True      # For symmetric tensors only
        },{
            'symbol': 'dt',
            'tensor_rank': 0,
            'positive': True
        },{
            'symbol': 'mu',
            'latex': '\\mu',
            'tensor_rank': 0,
            'positive': True
        },{
            'symbol': 'h',
            'tensor_rank': 0,
            'positive': True
        },{
            'symbol': 'dyn_tau',
            'latex': '\\tau_{dyn}',
            'tensor_rank': 0,
            'positive': True
        },{
            'symbol': 'stab_c1',
            'latex': 'c_{1\\,stab}',
            'tensor_rank': 0,
            'positive': True
        },{
            'symbol': 'stab_c2',
            'latex': 'c_{2\\,stab}',
            'tensor_rank': 0,
            'positive': True
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
    
    if artificial_compressibility:
        additional_nodal_tensors.extend([
        {
            'symbol': 'pn',
            'latex': 'p_n',
            'tensor_rank': 0
        },{
            'symbol': 'pnn',
            'latex': 'p_{nn}',
            'tensor_rank': 0
        },{
            'symbol': 'rho',
            'latex': '\\rho',
            'tensor_rank': 0
        },{
            'symbol': 'c',
            'tensor_rank': 0
        }
        ])
    else:
        additional_symbolic_tensors.extend([
        {
            'symbol': 'rho',
            'latex': '\\rho',
            'tensor_rank': 0
        }
        ])

    if darcy_term:
        additional_symbolic_tensors.extend([
        {
            'symbol': 'sigma',
            'latex': '\\sigma',
            'tensor_rank': 0,
            'positive': True
        },{
            'symbol': 'stab_c3',
            'latex': 'c_{3\\,stab}',
            'tensor_rank': 0,
            'positive': True
        }])
    else:
        numerical_tensors.extend([
        {
            'symbol': 'sigma',
            'latex': '\\sigma',
            'tensor_rank': 0,
            'value': 0.0
        },{
            'symbol': 'stab_c3',
            'latex': 'c_{3\\,stab}',
            'tensor_rank': 0,
            'value': 0.0
        }])

    if convective_term:
        if linearisation == "Picard":
            additional_nodal_tensors.extend([
            {
                'symbol': 'vconv',
                'latex': 'v_{conv}',
                'tensor_rank': 1
            }])
        elif linearisation == "FullNR":
            additional_nodal_tensors.extend([
            {
                'symbol': 'vmesh',
                'latex': 'v_{mesh}',
                'tensor_rank': 1
            }])
        else:
            raise Exception("Wrong linearisation \'" + linearisation + "\' selected. Available options are \'Picard\' and \'FullNR\'.")

    defined_functions = []

    expr = OrderedDict()

    expr['tau1_denom_aux1'] = "¨rho¨*¨dyn_tau¨/¨dt¨"
    expr['tau1_denom_aux2'] = "¨stab_c1¨*¨mu¨/¨h¨**2"
    expr['tau1_denom_aux3'] = "¨stab_c3¨*¨sigma¨/¨h¨"
    expr['tau2_aux1'] = "¨stab_c3¨*¨sigma¨/¨stab_c1¨"
    if convective_term:
        if (linearisation == "FullNR"):
            expr['vconv'] = "sub_op(¨v¨,¨vmesh¨)"
        expr['stab_norm_a'] = "norm(¨vconv¨)"
        expr['tau1_denom_aux4'] = "¨stab_c2¨*¨rho¨*¨stab_norm_a¨/¨h¨"
        expr['tau1_denom'] = "add_op(¨tau1_denom_aux1¨,¨tau1_denom_aux2¨,¨tau1_denom_aux3¨,¨tau1_denom_aux4¨)"
        expr['tau2_aux2'] = "¨stab_c2¨*¨rho¨*¨stab_norm_a¨*¨h¨/¨stab_c1¨"
        expr['tau2'] = "add_op(¨mu¨,¨tau2_aux1¨,¨tau2_aux2¨)"
    else:
        expr['tau1_denom'] = "add_op(¨tau1_denom_aux1¨,¨tau1_denom_aux2¨,¨tau1_denom_aux3¨)"
        expr['tau2'] = "add_op(¨mu¨,¨tau2_aux1¨)"
    expr['tau1'] = "1/¨tau1_denom¨"

    expr['accel'] = "add_op(prod_op(¨bdf0¨,¨v¨),prod_op(¨bdf1¨,¨vn¨),prod_op(¨bdf2¨,¨vnn¨))"

    if artificial_compressibility:
        expr['pder'] = "add_op(¨bdf0¨*¨p¨,¨bdf1¨*¨pn¨,¨bdf2¨*¨pnn¨)"

    if convective_term:
        expr['convective_term'] = f"contract(¨vconv¨,grad(¨v¨,¨base_scalars¨,{trans_str}))"
        expr['rho_convective_term'] = f"dot(¨vconv¨,grad(¨rho¨,¨base_scalars¨,{trans_str}))"

    expr['functional_aux1'] = "¨rho¨*dot(¨w¨,¨f¨)"
    expr['functional_aux2'] = "-1*¨rho¨*dot(¨w¨,¨accel¨)"
    expr['functional_aux3'] = f"-1*double_contract(grad(¨w¨,¨base_scalars¨,{trans_str}),¨stress¨)"
    expr['functional_aux4'] = "div(¨w¨,¨base_scalars¨)*¨p¨"
    if divide_by_rho:
        expr['functional_aux5'] = "-1*¨sigma¨*dot(¨w¨,sub_op(¨v¨,¨v_sol_frac¨))"
        expr['functional_aux6'] = "-1*div(¨v¨,¨base_scalars¨)*¨q¨"
        functional_tmp = "add_op(¨functional_aux1¨,¨functional_aux2¨,¨functional_aux3¨,¨functional_aux4¨,¨functional_aux5¨,¨functional_aux6¨)"
        if artificial_compressibility:
            functional_tmp = "sub_op("+functional_tmp+",1/(¨rho¨*¨c¨**2)*¨q¨*¨pder¨)"
            if convective_term:
                functional_tmp = "sub_op("+functional_tmp+",1/(¨rho¨)*¨q¨*¨rho_convective_term¨)"
    else:
        expr['functional_aux7'] = "-1*¨rho¨*¨q¨*div(¨v¨,¨base_scalars¨)"
        functional_tmp = "add_op(¨functional_aux1¨,¨functional_aux2¨,¨functional_aux3¨,¨functional_aux4¨,¨functional_aux7¨)"
        if artificial_compressibility:
            functional_tmp = "sub_op("+functional_tmp+",1/(¨c¨**2)*¨q¨*¨pder¨)"
            if convective_term:
                functional_tmp = "sub_op("+functional_tmp+",¨q¨*¨rho_convective_term¨)"
    if convective_term:
        functional_tmp = "sub_op("+functional_tmp+",¨rho¨*dot(¨w¨,¨convective_term¨))"
    expr['functional'] = functional_tmp

    expr['vel_residual_aux1'] = "prod_op(¨rho¨,¨f¨)"
    expr['vel_residual_aux2'] = "prod_op(-1*¨rho¨,¨accel¨)"
    expr['vel_residual_aux3'] = f"prod_op(-1,grad(¨p¨,¨base_scalars¨,{trans_str}))"
    expr['vel_residual_aux4'] = "prod_op(-1*¨sigma¨,sub_op(¨v¨,¨v_sol_frac¨))"
    vel_residual_tmp = "add_op(¨vel_residual_aux1¨,¨vel_residual_aux2¨,¨vel_residual_aux3¨,¨vel_residual_aux4¨)"
    if convective_term:
        vel_residual_tmp = "sub_op("+vel_residual_tmp+",prod_op(¨rho¨,¨convective_term¨))"
    expr['vel_residual'] = vel_residual_tmp

    if divide_by_rho:
        mas_residual_tmp = "-1*div(¨v¨,¨base_scalars¨)"
        if artificial_compressibility:
            mas_residual_tmp = "sub_op("+mas_residual_tmp+",1/(¨rho¨*¨c¨**2)*¨pder¨)"
            if convective_term:
                mas_residual_tmp = "sub_op("+mas_residual_tmp+",1/(¨rho¨)*¨rho_convective_term¨)"
    else:
        mas_residual_tmp = "-1*¨rho¨*div(¨v¨,¨base_scalars¨)"
        if artificial_compressibility:
            mas_residual_tmp = "sub_op("+mas_residual_tmp+",1/(¨c¨**2)*¨pder¨)"
            if convective_term:
                mas_residual_tmp = "sub_op("+mas_residual_tmp+",¨rho_convective_term¨)"
    expr['mas_residual'] = mas_residual_tmp

    expr['vel_subscale'] = "prod_op(¨tau1¨,¨vel_residual¨)"
    expr['mas_subscale'] = "¨tau2¨*¨mas_residual¨"

    functional_stab_tmp = f"dot(grad(¨q¨,¨base_scalars¨,{trans_str}),¨vel_subscale¨)"
    if not divide_by_rho:
        functional_stab_tmp = "prod_op(¨rho¨,"+functional_stab_tmp+")"
    if convective_term:
        expr['functional_stab_aux1'] = f"contract(¨vconv¨,grad(¨w¨,¨base_scalars¨,{trans_str}))"
        functional_stab_tmp = "add_op("+functional_stab_tmp+",¨rho¨*dot(¨functional_stab_aux1¨,¨vel_subscale¨))"
        functional_stab_tmp = "add_op("+functional_stab_tmp+",¨rho¨*div(¨vconv¨,¨base_scalars¨)*dot(¨w¨,¨vel_subscale¨))"
    functional_stab_tmp = "sub_op("+functional_stab_tmp+",¨sigma¨*dot(¨w¨,¨vel_subscale¨))"
    functional_stab_tmp = "add_op("+functional_stab_tmp+",div(¨w¨,¨base_scalars¨)*¨mas_subscale¨)"
    expr['functional_stab'] = functional_stab_tmp

    if ASGS_stabilization:
        expr['functional_final'] = "add_op(¨functional¨,¨functional_stab¨)"
    else:
        expr['functional_final'] = "¨functional¨"

    expr['explicit_stress'] = "double_contract(¨C¨,symgrad(¨v¨,¨base_scalars¨))"

    functional_rhs_name = 'functional_final'
    functional_lhs_name = 'functional_final'

    pre_lhs_substitutions = [('stress','explicit_stress')]
    post_lhs_substitutions = []

    return compile(dim, nnodes, use_transposed_gradients, unknown_vars, additional_nodal_tensors, additional_function_tensors, additional_symbolic_tensors,
              numerical_tensors, defined_functions, expr, functional_rhs_name, functional_lhs_name, pre_lhs_substitutions, post_lhs_substitutions, dofs)

