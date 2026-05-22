import json

import sympy as sp

from lib.tensor_placeholders import *
from lib.basic_classes import VarsCombination, CoefficientsIndicator
from lib.kratos_utilities import Compute_RHS, Compute_LHS
from lib.cpp_output_utils import OutputVector_CollectingFactors, OutputMatrix_CollectingFactors
from lib.printers import print_my_latex
from lib.components_utils import get_flat_list_of_components
from lib.utilities import substitute_all_placeholders_in_expression, substitute_all_arrays_in_expression
from lib.cpp_interface_dict import generate_interface_json
from lib.coordinates_system import CoordinateSystem
# from lib.placeholder_operators import *
from lib.dsl_interpreter import DSLInterpreter, SubstitutionsDSLInterpreter


class KratosFECompiler():

    def __init__(self, case_input_path):
        with open(case_input_path, 'r') as case_input_file:
            self.case_input = json.load(case_input_file)
        self.namespace={}

    def compile(self):
        nnodes = 3
        dim = 2
        impose_partion_of_unity = False
        transpose_gradients_flag = False

        # Define base scalars that will represent our reference coordinates system
        self.namespace['x'] = sp.Symbol('x')
        self.namespace['y'] = sp.Symbol('y')
        if dim==2:
            base_scalars = VarsCombination.from_namespace(['x','y'],self.namespace)
        elif dim==3:
            self.namespace['z'] = sp.Symbol('z')
            base_scalars = VarsCombination.from_namespace(['x','y','z'],self.namespace)

        # Define coordinates system (this will generate matrix for shape functions and their derivatives)
        material_coords_system = CoordinateSystem(base_scalars, nnodes, impose_partion_of_unity, transpose_gradients_flag)

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
            self.namespace[name] = UnknownTensorPlaceholder(trial_function_dict)
            self.namespace[test_name] = UnknownTensorPlaceholder(test_function_dict)
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
            functional_lhs = self.namespace["functional_lhs"]

            ## Print in high-level latex form
            # functional_evaluated = functional_rhs.doit(deep=True)
            latex_out_cont_compact = print_my_latex(functional_rhs)
            print(latex_out_cont_compact)
            print(functional_rhs)
    
            # Make array substitutions to the RHS functional
            functional_rhs = substitute_all_placeholders_in_expression(functional_rhs, placeholder_names_list, self.namespace)
            functional_lhs = substitute_all_placeholders_in_expression(functional_lhs, placeholder_names_list, self.namespace)
            print(functional_rhs)

            functional_rhs = functional_rhs.evaluate()
            functional_lhs = functional_lhs.evaluate()
            # print(sp.simplify(functional_rhs))
            # print(functional_rhs.evaluate())
            continuous_out = str(functional_rhs)
            latex_out_cont_expanded = sp.latex(functional_rhs)
            # print('RHS functional, array form:')
            print("RHS continuous array form")
            print(continuous_out)
            print(latex_out_cont_expanded)

            # Make array substitutions to any other expression that will be used later on.
            subst_dsl_interpreter = SubstitutionsDSLInterpreter(self.case_input['substitutions'])
            substitutions_dict = subst_dsl_interpreter.interpret_substitutions()
            pre_lhs_substitutions = substitutions_dict['pre_lhs']
            post_lhs_substitutions = substitutions_dict['post_lhs']
            print(substitutions_dict)
            
            # Make array substitutions to the LHS functional
            for subst_tuple in pre_lhs_substitutions:
                original_iterator = self.get_substitution_iterator_pre_lhs(subst_tuple[0], placeholder_names_list)
                new_iterator = self.get_substitution_iterator_pre_lhs(subst_tuple[1], placeholder_names_list)
                assert len(original_iterator)==len(new_iterator)
                for i in range(len(original_iterator)):
                    functional_lhs = functional_lhs.subs(original_iterator[i],new_iterator[i])

            print("LHS continuous array form")
            print(str(functional_lhs))


            ## Perform discretization
            functional_rhs = substitute_all_arrays_in_expression(functional_rhs, placeholder_names_list, self.namespace)
            gauss_out = str(functional_rhs)
            latex_out_gauss = sp.latex(functional_rhs)
            print('RHS gauss form')
            print(gauss_out)
            print(latex_out_gauss)

            functional_lhs = substitute_all_arrays_in_expression(functional_lhs, placeholder_names_list, self.namespace)
            print('LHS gauss form')
            print(str(functional_lhs))

            trial_symbols_list = []
            test_symbols_list = []
            for unknown_var in unknown_vars:
                trial_symbols_list.append(unknown_var['symbol'])
                test_symbols_list.append(unknown_var['test_function_symbol'])
            dofs = CoefficientsIndicator.from_namespace(trial_symbols_list, self.namespace)
            dofs = sp.Matrix(dofs.discretize_expression(placeholder_names_list, self.namespace))
            testfunc = CoefficientsIndicator.from_namespace(test_symbols_list, self.namespace)
            testfunc = sp.Matrix(testfunc.discretize_expression(placeholder_names_list, self.namespace))
            print("DOFS")
            print(dofs)
            print("TEST FUNCTIONS")
            print(testfunc)
    
            do_simplifications = False
            rhs = Compute_RHS(functional_rhs.copy(), testfunc, do_simplifications)
            w_g = sp.Symbol("w_g", positive=True) # Gauss point integration weight
            mode="c"
            rhs_out = OutputVector_CollectingFactors(w_g*rhs, "rRightHandSideVector", mode, indentation_level=2, assignment_op="+=")
            print(rhs_out)

            pre_lhs = Compute_RHS(functional_lhs.copy(), testfunc, do_simplifications)
            lhs = Compute_LHS(pre_lhs, testfunc, dofs, do_simplifications)
            print("LHS matrix form before substitutions:")
            print(str(lhs))

            for subst_tuple in post_lhs_substitutions:
                original_iterator = self.get_substitution_iterator_post_lhs(subst_tuple[0], placeholder_names_list)
                new_iterator = self.get_substitution_iterator_post_lhs(subst_tuple[1], placeholder_names_list)
                assert len(original_iterator)==len(new_iterator)
                for i in range(len(original_iterator)):
                    lhs = lhs.subs(original_iterator[i],new_iterator[i])
                print("SUBSTITUTION "+str(original_iterator)+" with "+str(new_iterator))
            
            print("LHS matrix form after substitutions:")
            print(str(lhs))

    def get_substitution_iterator_pre_lhs(self, component, placeholder_names_list):
        if isinstance(component, DerivIndicator):
            iterator = component.get_derivative_iterator_pre_lhs(placeholder_names_list, self.namespace)
        elif isinstance(component, str):
            auxiliary_expr = substitute_all_placeholders_in_expression(self.namespace[component], placeholder_names_list, self.namespace)
            if not isinstance(auxiliary_expr,sp.NDimArray):
                auxiliary_expr = auxiliary_expr.evaluate()
            iterator = get_flat_list_of_components(auxiliary_expr)
        else:
            raise "Substitution tuples should contain only strings or DerivIndicators.  Got "+str(type(component))+" for component "+str(component)+". "
        return iterator
    
    def get_substitution_iterator_post_lhs(self, component, placeholder_names_list):
        if isinstance(component, DerivIndicator):
            iterator = component.get_derivative_iterator_post_lhs(placeholder_names_list, self.namespace)
        elif isinstance(component, str):
            expression = substitute_all_placeholders_in_expression(self.namespace[component], placeholder_names_list, self.namespace)
            if not (isinstance(expression, sp.NDimArray) or isinstance(expression, sp.Function)):
                expression = expression.evaluate()
            expression = substitute_all_arrays_in_expression(expression, placeholder_names_list, self.namespace)
            iterator = get_flat_list_of_components(expression)
        else:
            raise "Substitution tuples should contain only strings or DerivIndicators"
        return iterator

if __name__=="__main__":

    compiler = KratosFECompiler('test_input.json')
    compiler.compile()