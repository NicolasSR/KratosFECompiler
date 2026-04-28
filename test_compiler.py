import json

import sympy as sp

from lib.tensor_placeholders import *
from lib.basic_classes import VarsCombination, DofsIndicator
from lib.kratos_utilities import Compute_RHS, Compute_LHS
from lib.cpp_output_utils import OutputVector_CollectingFactors, OutputMatrix_CollectingFactors
from lib.printers import print_my_latex
from lib.components_utils import get_flat_list_of_components, get_tuple_subelements
from lib.utilities import substitute_symbols, substitute_functions
from lib.cpp_interface_dict import generate_interface_json
from lib.coordinates_system import CoordinateSystem
# from lib.placeholder_operators import *
from lib.dsl_interpreter import DSLInterpreter


class KratosFECompiler():

    def __init__(self, case_input_path):
        with open(case_input_path, 'r') as case_input_file:
            self.case_input = json.load(case_input_file)
        self.namespace={}

    def compile(self):
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

        ## Get quantities of the PDE

        unknown_vars = self.case_input['quantities']['unknown_vars']

        # Assign new TensorPlaceholders to all variables
        placeholder_names_list = []
        for var in unknown_vars:
            name = var['symbol']
            test_name = var['test_function_symbol']
            rank = var['tensor_rank']
            trial_function_dict = {'symbol': name, 'latex':var.get('latex', name),
                                    'tensor_rank': rank, 'dim':[dim]*rank, 'dependencies': base_scalars}
            test_function_dict = {'symbol': test_name, 'latex':var.get('test_function_latex', test_name),
                                    'tensor_rank': rank, 'dim':[dim]*rank, 'dependencies': base_scalars}
            self.namespace[name] = BaseTensorPlaceholder(trial_function_dict)
            self.namespace[test_name] = BaseTensorPlaceholder(test_function_dict)
            placeholder_names_list.append(name)
            placeholder_names_list.append(test_name)

        # We apply our coordinate system
        with material_coords_system:

            # Code for the functional is built from SSA (Static Single Assignment)-based AST in JSON format

            functional_ssa = self.case_input['functional_ssa']
            dsl_interpreter = DSLInterpreter(functional_ssa, self.namespace)
            dsl_interpreter.interpret_ssa()

            print(self.namespace)

            functional_rhs = self.namespace["functional_rhs"]

            ## Print in high-level latex form
            # functional_evaluated = functional_rhs.doit(deep=True)
            latex_out_cont_compact = print_my_latex(functional_rhs)
            print(latex_out_cont_compact)
            print(functional_rhs)

            # Define all array-type variables from tensor placeholders
            for name in placeholder_names_list:
                self.namespace[name].generate_array()

            def substitute_all_placeholders(expr):
                subst_list = [( self.namespace[name],  self.namespace[name].array) for name in placeholder_names_list] 
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


if __name__=="__main__":

    compiler = KratosFECompiler('test_input.json')
    compiler.compile()