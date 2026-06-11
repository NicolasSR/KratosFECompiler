import json

import sympy as sp

from lib.tensor_placeholders import *
from lib.basic_classes import CompilerNamespace, VarsCombination, CoefficientsIndicator
from lib.kratos_utilities import Compute_RHS, Compute_LHS
from lib.cpp_output_utils import OutputVector_CollectingFactors, OutputMatrix_CollectingFactors
from lib.printers import print_my_latex
from lib.substitution_utils import substitute_all_placeholders_in_expression, substitute_all_arrays_in_expression, get_substitution_iterator_pre_lhs, get_substitution_iterator_post_lhs, substitute_defined_functions_in_expression
from lib.cpp_interface_dict import generate_interface_json
from lib.coordinates_system import CoordinateSystem
from lib.dsl_interpreter import DSLInterpreter, SubstitutionsDSLInterpreter


class KratosFECompiler():

    def __init__(self, case_input_path):
        with open(case_input_path, 'r') as case_input_file:
            self.case_input = json.load(case_input_file)
        self.namespace=CompilerNamespace()

    def compile(self, options_dict):

        ## Symbolic generation settings
        dim = options_dict["dim"]
        nnodes = options_dict["nnodes"]

        # Get the configuration settings from the case_config.json (priority) or from the fe_definition.json file (fallback)
        config_settings_dict = {}
        for config_setting in self.case_input["config_settings"]:
            if config_setting in options_dict:
                config_settings_dict[config_setting] = options_dict[config_setting]
            else:
                config_settings_dict[config_setting] = self.case_input["config_settings"][config_setting]

        impose_partion_of_unity = False
        # Get the flag to transpose gradients from the fe_definition.json file, default is False
        transpose_gradients_flag = self.case_input["config_settings"].get("transpose_gradients_flag", False)

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
        nodal_vars = self.case_input['quantities'].get('nodal_vars',[])
        symbolic_vars = self.case_input['quantities'].get('symbolic_vars',[])
        constant_vars = self.case_input['quantities'].get('constant_vars',[])
        numerical_vars = self.case_input['quantities'].get('numerical_vars',[])
        undefined_functions = self.case_input['quantities'].get('undefined_functions',[])
        defined_functions = self.case_input['quantities'].get('defined_functions',[])

        # Assign new TensorPlaceholders to all variables
        placeholder_names_list = []
        for var in unknown_vars:
            name = var['symbol']
            test_name = var['test_function_symbol']
            rank = int(var['tensor_rank'])
            trial_function_dict = {'symbol': name, 'latex':var.get('latex', name),
                                    'tensor_rank': rank, 'dim':[dim]*rank, 'dependencies': base_scalars}
            test_function_dict = {'symbol': test_name, 'latex':var.get('test_function_latex', test_name),
                                    'tensor_rank': rank, 'dim':[dim]*rank, 'dependencies': base_scalars}
            self.namespace[name] = UnknownTensorPlaceholder.from_info_dict(trial_function_dict)
            self.namespace[test_name] = UnknownTensorPlaceholder.from_info_dict(test_function_dict)
            placeholder_names_list.append(name)
            placeholder_names_list.append(test_name)

        trial_symbols_list = []
        test_symbols_list = []
        for unknown_var in unknown_vars:
            trial_symbols_list.append(unknown_var['symbol'])
            test_symbols_list.append(unknown_var['test_function_symbol'])
        dofs = CoefficientsIndicator.from_namespace(trial_symbols_list, self.namespace)
        self.namespace['dofs'] = dofs
        testfunc = CoefficientsIndicator.from_namespace(test_symbols_list, self.namespace)

        for var in nodal_vars:
            name = var['symbol']
            rank = int(var['tensor_rank'])
            self.namespace[name] = NodalTensorPlaceholder.from_info_dict({'symbol': name, 'latex':var.get('latex', name),
                                    'tensor_rank': rank, 'dim':[dim]*rank, 'dependencies': base_scalars,
                                    'positive': var.get('positive', False) })
            placeholder_names_list.append(name)

        for var in symbolic_vars:
            name = var['symbol']
            rank = int(var['tensor_rank'])
            self.namespace[name] = SymbolicTensorPlaceholder.from_info_dict({'symbol': name, 'latex':var.get('latex', name),
                                    'tensor_rank': rank, 'dim':[dim]*rank, 'dependencies': base_scalars,
                                    'symmetric': var.get('symmetric', False), 'third_symmetry': var.get('third_symmetry', False),
                                    'use_voigt_notation': var.get('use_voigt_notation', False), "positive": var.get("positive", False) })
            placeholder_names_list.append(name)

        for var in constant_vars:
            name = var['symbol']
            rank = int(var['tensor_rank'])
            self.namespace[name] = ConstantTensorPlaceholder.from_info_dict({'symbol': name, 'latex':var.get('latex', name),
                                    'tensor_rank': rank, 'dim':[dim]*rank, 'dependencies': "",
                                    'symmetric': var.get('symmetric', False), 'third_symmetry': var.get('third_symmetry', False),
                                    'use_voigt_notation': var.get('use_voigt_notation', False), "positive": var.get("positive", False) })
            placeholder_names_list.append(name)

        for var in numerical_vars:
            name = var['symbol']
            rank = int(var['tensor_rank'])
            self.namespace[name] = NumericalTensorPlaceholder.from_info_dict({'symbol': name, 'latex':var.get('latex', name),
                                    'tensor_rank': rank, 'dim':[dim]*rank, 'dependencies': var['value']})
            placeholder_names_list.append(name)

        undefined_functions_list = []
        for var in undefined_functions:
            name = var['symbol']
            rank = int(var['tensor_rank'])
            self.namespace[name] = UndefinedFunctionTensorPlaceholder.from_info_dict({'symbol': name, 'latex':var.get('latex', name),
                                    'tensor_rank': rank, 'dim':[dim]*rank, 'dependencies': var['dependencies']}, self.namespace)
            undefined_functions_list.append(name)

        defined_functions_list = []
        for var in defined_functions:
            name = var['symbol']
            rank = int(var['tensor_rank'])
            self.namespace[name] = DefinedFunctionTensorPlaceholder.from_info_dict({'symbol': name, 'latex':var.get('latex', name),
                                    'tensor_rank': rank, 'dim':[dim]*rank, 'dependencies': ""})
            defined_functions_list.append(name)

        # We apply our coordinate system
        with material_coords_system:

            # Code for the functional is built from SSA (Static Single Assignment)-based AST in JSON format

            functional_ssa = self.case_input['functional_ssa']
            dsl_interpreter = DSLInterpreter(self.namespace, defined_functions_list, config_settings_dict)
            dsl_interpreter.interpret_ssa(functional_ssa)

            print(self.namespace)

            functional_rhs = self.namespace["functional_rhs"]
            if "functional_lhs" in self.namespace.keys():
                functional_lhs = self.namespace["functional_lhs"]
            else:
                functional_lhs = functional_rhs.copy()

            ## Print in high-level latex form
            print("Functional in continuous, compact form:")
            latex_out_cont_compact = print_my_latex(functional_rhs)
            print(latex_out_cont_compact)
            print(functional_rhs)

            functional_rhs = substitute_defined_functions_in_expression(functional_rhs, defined_functions_list[::-1], self.namespace)
            functional_lhs = substitute_defined_functions_in_expression(functional_lhs, defined_functions_list[::-1], self.namespace)

            print("Functional in continuous, complete form after substituting predefined functions:")
            latex_out_cont_compact_2 = print_my_latex(functional_rhs)
            print(latex_out_cont_compact_2)
            print(functional_rhs)

            # Make array substitutions to the RHS functional
            functional_rhs = substitute_all_placeholders_in_expression(functional_rhs, placeholder_names_list+undefined_functions_list, self.namespace)
            functional_lhs = substitute_all_placeholders_in_expression(functional_lhs, placeholder_names_list+undefined_functions_list, self.namespace)

            # First the symbolic/nodal var placeholders

            print("Functional in continuous, expanded form after substituting placeholders:")
            print(functional_rhs)

            functional_rhs = functional_rhs.evaluate()
            functional_lhs = functional_lhs.evaluate()

            continuous_out = str(functional_rhs)
            latex_out_cont_expanded = sp.latex(functional_rhs)
            print("RHS continuous array form")
            print(latex_out_cont_expanded)
            print(continuous_out)

            # Get substitution tuples defined in the case definition json
            subst_dsl_interpreter = SubstitutionsDSLInterpreter(self.case_input['substitutions'], self.namespace)
            substitutions_dict = subst_dsl_interpreter.interpret_substitutions()
            pre_lhs_substitutions = substitutions_dict.get('pre_lhs',[])
            post_lhs_substitutions = substitutions_dict.get('post_lhs',[])
            print(substitutions_dict)
            
            # Make array substitutions to the LHS functional
            for subst_tuple in pre_lhs_substitutions:
                original_iterator = get_substitution_iterator_pre_lhs(subst_tuple[0], placeholder_names_list+undefined_functions_list, defined_functions_list, self.namespace)
                new_iterator = get_substitution_iterator_pre_lhs(subst_tuple[1], placeholder_names_list+undefined_functions_list, defined_functions_list, self.namespace)
                assert len(original_iterator)==len(new_iterator)
                for i in range(len(original_iterator)):
                    functional_lhs = functional_lhs.subs(original_iterator[i],new_iterator[i])

            print("LHS continuous array form")
            print(str(functional_lhs))

            ## Perform discretization
            functional_rhs = substitute_all_arrays_in_expression(functional_rhs, placeholder_names_list+undefined_functions_list, self.namespace)
            gauss_out = str(functional_rhs)
            latex_out_gauss = sp.latex(functional_rhs)
            print('RHS gauss form')
            print(gauss_out)
            print(latex_out_gauss)

            functional_lhs = substitute_all_arrays_in_expression(functional_lhs, placeholder_names_list+undefined_functions_list, self.namespace)
            print('LHS gauss form')
            print(str(functional_lhs))

            dofs_as_matrix = sp.Matrix(dofs.gauss)
            testfunc_as_matrix = sp.Matrix(testfunc.gauss)
            print("DOFS")
            print(dofs_as_matrix)
            print("TEST FUNCTIONS")
            print(testfunc_as_matrix)
    
            do_simplifications = False
            rhs = Compute_RHS(functional_rhs.copy(), testfunc_as_matrix, do_simplifications)
            w_g = sp.Symbol("w_g", positive=True) # Gauss point integration weight
            mode="c"
            rhs_out = OutputVector_CollectingFactors(w_g*rhs, "rRightHandSideVector", mode, indentation_level=2, assignment_op="+=")
            print(rhs_out)

            pre_lhs = Compute_RHS(functional_lhs.copy(), testfunc_as_matrix, do_simplifications)
            lhs = Compute_LHS(pre_lhs, testfunc_as_matrix, dofs_as_matrix, do_simplifications)
            print("LHS matrix form before substitutions:")
            print(str(lhs))

            for subst_tuple in post_lhs_substitutions:
                original_iterator = get_substitution_iterator_post_lhs(subst_tuple[0], placeholder_names_list+undefined_functions_list, defined_functions_list[::-1], self.namespace)
                new_iterator = get_substitution_iterator_post_lhs(subst_tuple[1], placeholder_names_list+undefined_functions_list, defined_functions_list[::-1], self.namespace)
                assert len(original_iterator)==len(new_iterator)
                for i in range(len(original_iterator)):
                    lhs = lhs.subs(original_iterator[i],new_iterator[i])
            
            print("LHS matrix form after substitutions:")
            print(str(lhs))

            lhs_out = OutputMatrix_CollectingFactors(w_g*lhs, "rLeftHandSideMatrix", mode, indentation_level=2, assignment_op="+=")
    
            full_output_string = f"Functional in continuous, compact form:\n$${latex_out_cont_compact}$$\n\nFunctional in continuous, compact form after substituting predefined functions:\n$${latex_out_cont_compact_2}$$\n\nFunctional in continuous, expanded form:\n$${latex_out_cont_expanded}$$\n\nFunctional in discretized from:\n$${latex_out_gauss}$$"

            cpp_interface_json = generate_interface_json(dim, nnodes, placeholder_names_list, self.namespace, sp.shape(dofs_as_matrix)[0])

            return rhs_out, lhs_out, full_output_string, cpp_interface_json

if __name__=="__main__":

    compiler = KratosFECompiler('test_input.json')
    compiler.compile()