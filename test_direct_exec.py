import re
from collections import OrderedDict

import sympy as sp

from lib.tensor_placeholders import *
from lib.basic_classes import VarsCombination, DofsIndicator
from lib.kratos_utilities import DefineShapeFunctions, Compute_RHS, Compute_LHS
from lib.cpp_output_utils import OutputVector_CollectingFactors, OutputMatrix_CollectingFactors
from lib.printers import print_my_latex
from lib.components_utils import get_flat_list_of_components, get_tuple_subelements
from lib.utilities import substitute_symbols, substitute_functions
from lib.cpp_interface_dict import generate_interface_json
from lib.coordinates_system import CoordinateSystem
from lib.placeholder_operators import *

nnodes = 3
dim = 2
impose_partion_of_unity = False

# Define base scalars that will represent our reference coordinates system
x = sp.Symbol('x')
y = sp.Symbol('y')
if dim==2:
    base_scalars = VarsCombination(x,y)
elif dim==3:
    z = sp.Symbol('z')
    base_scalars = VarsCombination(x,y,z)

# Define coordinates system (this will generate matrix for shape functions and their derivatives)
material_coords_system = CoordinateSystem(base_scalars, nnodes, impose_partion_of_unity)

## Define variables of the PDE

# Initialize to None
v = None
w = None
p = None
q = None

# Determine variable properties as list of dicts
unknown_vars = [        # For each unknown var we need symbol, tensor rank and symbol for the corresponding test function
    {
        'symbol': 'v',
        'tensor_rank': 1,
        'test_function_symbol': 'w'
    },{
        'symbol': 'p',
        'tensor_rank': 0,
        'test_function_symbol': 'q'
    }
]

# Assign new TensorPlaceholders to all variables
placeholders_list = []
for var in unknown_vars:
    name = var['symbol']
    test_name = var['test_function_symbol']
    rank = var['tensor_rank']
    trial_function_dict = {'symbol': name, 'latex':var.get('latex', name),
                            'tensor_rank': rank, 'dim':[dim]*rank, 'dependencies': base_scalars}
    test_function_dict = {'symbol': test_name, 'latex':var.get('test_function_latex', test_name),
                            'tensor_rank': rank, 'dim':[dim]*rank, 'dependencies': base_scalars}
    locals()[name] = BaseTensorPlaceholder(trial_function_dict)
    locals()[test_name] = BaseTensorPlaceholder(test_function_dict)
    placeholders_list.append(locals()[name])
    placeholders_list.append(locals()[test_name])

# We apply our coordinate system
with material_coords_system:

    ## Write PDE expression
    aux_1 = doublecontract_op(grad_op(v),grad_op(w))
    aux_2 = p+norm_op(w)+2
    functional_rhs = aux_1+aux_2

    # functional_rhs = grad_op(v)+grad_op(w)

    # functional_rhs = (norm_op(v)+contract_op(w,v))*w*q
    # f = norm_op(v)
    # functional_rhs = v*(p+3+f)+w+v+w
    # expr['functional'] = "(¨v¨*(¨p¨*3+¨f¨))+¨w¨+¨v¨+¨w¨"
    # expr['functional'] = "(norm(¨v¨)+contract(¨w¨,¨v¨))*¨w¨*¨q¨"

    ## Print in high-level latex form
    # functional_evaluated = functional_rhs.doit(deep=True)
    latex_out_cont_compact = print_my_latex(functional_rhs)
    print(latex_out_cont_compact)
    print(functional_rhs)

    # Define all array-type variables from tensor placeholders
    for placeholder in placeholders_list:
        placeholder.generate_array()

    def substitute_all_placeholders(expr):
        subst_list = [(placeholder, placeholder.array) for placeholder in placeholders_list] 
        expr = expr.subs(subst_list, evaluate=False)
        return expr

    # Make array substitutions to the RHS functional
    functional_rhs = substitute_all_placeholders(functional_rhs)
    print(functional_rhs)
    functional_rhs = functional_rhs.evaluate()

    # print(sp.simplify(functional_rhs))
    # print(functional_rhs.evaluate())

    continuous_out = str(functional_rhs)
    latex_out_cont_expanded = sp.latex(functional_rhs)
    # print('RHS functional, array form:')
    print(continuous_out)
    print(latex_out_cont_expanded)
