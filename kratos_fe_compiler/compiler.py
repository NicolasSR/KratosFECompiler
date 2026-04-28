import re

import sympy as sp

from lib.tensor_placeholders import *
from lib.basic_classes import VarsCombination, DofsIndicator, add_op, prod_op, sub_op, neg_op
from lib.kratos_utilities import DefineShapeFunctions, Compute_RHS, Compute_LHS
from lib.cpp_output_utils import OutputVector_CollectingFactors, OutputMatrix_CollectingFactors
from lib.printers import print_my_latex
from lib.components_utils import get_flat_list_of_components, get_tuple_subelements
from lib.utilities import substitute_symbols, substitute_functions
import lib.placeholder_operators as po
from lib.cpp_interface_dict import generate_interface_json

def compile(dim, nnodes, use_transposed_gradients, unknown_vars, additional_nodal_tensors, additional_function_tensors, additional_symbolic_tensors, numerical_tensors,
              defined_functions, expressions_ordered_dict, functional_rhs_name, functional_lhs_name, pre_lhs_substitutions, post_lhs_substitutions, dofs):

    impose_partion_of_unity = False
    N,DN = DefineShapeFunctions(nnodes, dim, impose_partion_of_unity)

    SYMB = {} # Dictionary where we will store all of our Sympy vars

    SYMB['x'] = sp.Symbol('x')
    SYMB['y'] = sp.Symbol('y')
    if dim==2:
        # base_scalars = sp.Array([SYMB['x'],SYMB['y']])
        base_scalars = VarsCombination('x','y')
    elif dim==3:
        SYMB['z'] = sp.Symbol('z')
        # base_scalars = sp.Array([SYMB['x'],SYMB['y'],SYMB['z']])
        base_scalars = VarsCombination('x','y','z')
    base_scalars.update_symbol_objects(SYMB)
    SYMB['base_scalars'] = base_scalars.get_flat_objects_list(SYMB,'')
    # SYMB['base_scalars'] = base_scalars

    placeholders_list = []
    defined_functions_list = []

    # Define tensor placeholders for unknown variables
    for unknown_var in unknown_vars:
        name = unknown_var['symbol']
        test_name = unknown_var['test_function_symbol']
        rank = unknown_var['tensor_rank']
        dependencies = disambiguate_var_group(base_scalars, SYMB)
        trial_function_dict = {'symbol': name, 'latex':unknown_var.get('latex', name),
                               'tensor_rank': rank, 'dependencies': dependencies,
                               'dim':dim, 'N':N, 'DN':DN
                            }
        test_function_dict = {'symbol': test_name, 'latex':unknown_var.get('test_function_latex', test_name),
                              'tensor_rank': rank, 'dependencies': dependencies,
                              'dim':dim, 'N':N, 'DN':DN}
        if rank == 0:
            SYMB[name] = UnknownTensorPlaceholderRank0(sp.Symbol(name))
            SYMB[test_name] = UnknownTensorPlaceholderRank0(sp.Symbol(test_name))
        elif rank == 1:
            SYMB[name] = UnknownTensorPlaceholderRank1(sp.Symbol(name))
            SYMB[test_name] = UnknownTensorPlaceholderRank1(sp.Symbol(test_name))
        else:
            raise "Unknown variables should be either scalar or vectorial"
        SYMB[name].apply_tensor_config(trial_function_dict)
        SYMB[test_name].apply_tensor_config(test_function_dict)
        placeholders_list.append(SYMB[name])
        placeholders_list.append(SYMB[test_name])

    for nodal_tensor in additional_nodal_tensors:
        name = nodal_tensor['symbol']
        rank = nodal_tensor['tensor_rank']
        nodal_tensor['dim']=dim
        nodal_tensor['dependencies']=disambiguate_var_group(base_scalars,SYMB)
        nodal_tensor['N']=N
        nodal_tensor['DN']=DN
        if rank == 0:
            SYMB[name] = UnknownTensorPlaceholderRank0(sp.Symbol(name))
        elif rank == 1:
            SYMB[name] = UnknownTensorPlaceholderRank1(sp.Symbol(name))
        else:
            raise "Nodal variables should be either scalar or vectorial"
        SYMB[name].apply_tensor_config(nodal_tensor)
        placeholders_list.append(SYMB[name])

    
    # Update the DoFs indicator to take the objects from SYMB
    dofs.update_symbol_objects(SYMB)

    # Define tensor placeholders for additional function tensors. Order is important because of possible dependencies.
    for function_tensor in additional_function_tensors:
        name = function_tensor['symbol']
        rank = function_tensor['tensor_rank']
        function_tensor['dim']=dim
        if rank == 0:
            SYMB[name] = FunctionTensorPlaceholderRank0(sp.Symbol(name))
        elif rank == 1:
            SYMB[name] = FunctionTensorPlaceholderRank1(sp.Symbol(name))
        elif rank == 2:
            SYMB[name] = FunctionTensorPlaceholderRank2(sp.Symbol(name))
        elif rank == 4:
            SYMB[name] = FunctionTensorPlaceholderRank4(sp.Symbol(name))
        else:
            raise "Function tensors must be of rank 0, 1, 2, or 4"
        function_tensor['dependencies'] = disambiguate_var_group(function_tensor['dependencies'],SYMB)
        SYMB[name].apply_tensor_config(function_tensor)
        placeholders_list.append(SYMB[name])
    
    # Define tensor placeholders for additional symbolic tensors (no explicit dependencies).
    for symbolic_tensor in additional_symbolic_tensors:
        name = symbolic_tensor['symbol']
        rank = symbolic_tensor['tensor_rank']
        symbolic_tensor['dim']=dim
        if rank == 0:
            SYMB[name] = SymbolicTensorPlaceholderRank0(sp.Symbol(name))
        elif rank == 1:
            SYMB[name] = SymbolicTensorPlaceholderRank1(sp.Symbol(name))
        elif rank == 2:
            SYMB[name] = SymbolicTensorPlaceholderRank2(sp.Symbol(name))
        elif rank == 4:
            SYMB[name] = SymbolicTensorPlaceholderRank4(sp.Symbol(name))
        else:
            raise "Symbolic tensors must be of rank 0, 1, 2 or 4"
        SYMB[name].apply_tensor_config(symbolic_tensor)
        placeholders_list.append(SYMB[name])
    
    # Define tensor placeholders for additional numerical tensors (to be substituted by fixed quantities).
    for numerical_tensor in numerical_tensors:
        name = numerical_tensor['symbol']
        numerical_tensor['dim']=dim
        rank = symbolic_tensor['tensor_rank']
        if rank == 0:
            SYMB[name] = NumericalTensorPlaceholderRank0(sp.Symbol(name))
        else:
            SYMB[name] = NumericalTensorPlaceholder(sp.Symbol(name))
        SYMB[name].apply_tensor_config(numerical_tensor)
        placeholders_list.append(SYMB[name])

    for defined_function in defined_functions:
        name = defined_function['symbol']
        defined_function['dim']=dim
        SYMB[name] = DefinedFunctionPlaceholder(sp.Symbol(name))
        SYMB[name].apply_tensor_config(defined_function)
        defined_functions_list.append(SYMB[name])


    # Execute all the expression definitions and save them into SYMB dictionary
    for expr_key, expr_value in expressions_ordered_dict.items():
        substituted_expr = substitute_symbols(expr_value)
        substituted_expr = substitute_functions(substituted_expr)
        # print(substituted_expr)
        print(substituted_expr)
        exec(f"SYMB['{expr_key}'] = {substituted_expr}")

    # Take the RHS functional from the expressions in SYMB
    functional_rhs = SYMB[functional_rhs_name].copy()
    functional_lhs = SYMB[functional_lhs_name].copy()

    ## Print in high-level latex form
    functional_evaluated = functional_rhs.doit(deep=True)
    latex_out_cont_compact = print_my_latex(functional_evaluated)
    print(latex_out_cont_compact)
    print(functional_evaluated)

    ## Substitute defined functions:
    for defined_function in defined_functions_list:
        substituted_expr = defined_function.generate_explicit_functions()
        exec(f"SYMB['{defined_function.name}'] = {substituted_expr}")

    # Execute all the expression definitions and save them into SYMB dictionary
    for expr_key, expr_value in expressions_ordered_dict.items():
        substituted_expr = substitute_symbols(expr_value)
        substituted_expr = substitute_functions(substituted_expr)
        print(substituted_expr)
        exec(f"SYMB['{expr_key}'] = {substituted_expr}")

    # Take the RHS functional from the expressions in SYMB
    functional_rhs = SYMB[functional_rhs_name].copy()
    functional_lhs = SYMB[functional_lhs_name].copy()

    ## Print in high-level latex form with defined functions substituted:
    functional_evaluated = functional_rhs.doit(deep=True)
    latex_out_cont_compact_2 = print_my_latex(functional_evaluated)
    print(latex_out_cont_compact_2)
    print(functional_evaluated)


    ## Substitution by actual Arrays

    SYMB['base_scalars_array'] = base_scalars.get_array()

    # Define all array-type variables from tensor placeholders
    for placeholder in placeholders_list:
        placeholder.generate_array(SYMB)

    def substitute_all_placeholders(expr):
        for placeholder in placeholders_list:
            expr = expr.subs(placeholder, placeholder.array)
        return expr

    # Make array substitutions to the RHS functional
    functional_rhs = substitute_all_placeholders(functional_rhs)

    continuous_out = str(functional_rhs)
    latex_out_cont_expanded = sp.latex(functional_rhs)
    # print('RHS functional, array form:')
    print(continuous_out)
    print(latex_out_cont_expanded)

    #Try simplify:
    # functional_rhs_simplified = sp.factor(functional_rhs.args[0]+functional_rhs.args[2]+functional_rhs.args[3])
    # functional_rhs_simplified = sp.collect(functional_rhs, SYMB['k_array'])
    # print(functional_rhs_simplified)
    # print(sp.latex(functional_rhs_simplified))

    # Make array substitutions to any other expression that will be used later on.
    for subst_tuple in pre_lhs_substitutions + post_lhs_substitutions:
        arrays_to_define = [arg_name for arg_name in get_tuple_subelements(subst_tuple) if not arg_name+'_array' in SYMB.keys()]
        for var_name in arrays_to_define:
            SYMB[var_name+'_array']=substitute_all_placeholders(SYMB[var_name].copy())

    # Make array substitutions to the LHS functional
    functional_lhs = substitute_all_placeholders(functional_lhs)
    for subst_tuple in pre_lhs_substitutions:
        if isinstance(subst_tuple[0], DerivIndicator):
            original_iterator = subst_tuple[0].get_derivative_iterator(SYMB, '_array', use_transposed_gradients)
        elif isinstance(subst_tuple[0], str):
            original_iterator = get_flat_list_of_components(SYMB[subst_tuple[0]+'_array'])
        else:
            raise "Substitution tuples should contain only strings or DerivIndicators"
        if isinstance(subst_tuple[1], DerivIndicator):
            new_iterator = subst_tuple[1].get_derivative_iterator(SYMB, '_array', use_transposed_gradients)
        elif isinstance(subst_tuple[1], str):
            new_iterator = get_flat_list_of_components(SYMB[subst_tuple[1]+'_array'])
        else:
            raise "Substitution tuples should contain only strings or DerivIndicators"
        assert len(original_iterator)==len(new_iterator)
        for i in range(len(original_iterator)):
            functional_lhs = functional_lhs.subs(original_iterator[i],new_iterator[i])
    # print('LHS functional, array form:')
    # print(functional_lhs)


    ## Perform discretization

    for placeholder in placeholders_list:
        placeholder.generate_gauss(SYMB)

    def substitute_all_placeholders_gauss(expr):
        for placeholder in reversed(placeholders_list):
            expr = placeholder.substitute_gauss(expr)
        return expr

    # Make array substitutions to the RHS functional
    functional_rhs = substitute_all_placeholders_gauss(functional_rhs)

    gauss_out = str(functional_rhs)
    latex_out_gauss = sp.latex(functional_rhs)    # print('RHS functional, gauss form:')
    print(gauss_out)
    print(latex_out_gauss)

    # Make array substitutions to the RHS functional
    functional_lhs = substitute_all_placeholders_gauss(functional_lhs)
    # print('LHS functional, gauss form:')
    # print(functional_lhs)

    trial_symbols_list = []
    test_symbols_list = []
    for unknown_var in unknown_vars:
        trial_symbols_list.append(unknown_var['symbol'])
        test_symbols_list.append(unknown_var['test_function_symbol'])
    dofs = sp.Matrix(DofsIndicator(*trial_symbols_list).get_dependency_list_array_or_matrix(SYMB,''))
    testfunc = sp.Matrix(DofsIndicator(*test_symbols_list).get_dependency_list_array_or_matrix(SYMB,''))

    do_simplifications = False
    rhs = Compute_RHS(functional_rhs.copy(), testfunc, do_simplifications)
    w_g = sp.Symbol("w_g", positive=True) # Gauss point integration weight
    mode="c"
    rhs_out = OutputVector_CollectingFactors(w_g*rhs, "rRightHandSideVector", mode, indentation_level=2, assignment_op="+=")

    print(rhs_out)

    pre_lhs = Compute_RHS(functional_lhs.copy(), testfunc, do_simplifications)
    lhs = Compute_LHS(pre_lhs, testfunc, dofs, do_simplifications)

    # Calculate the LHS
    # n_dofs = pre_lhs.shape[0]
    # lhs = sp.zeros(n_dofs, n_dofs)
    # for i in range(n_dofs):
    #     for j in range(n_dofs):
    #         lhs[i,j] -= sp.diff(pre_lhs[i], dofs[j])

    # Make array substitutions to the LHS
    for subst_tuple in post_lhs_substitutions:
        if isinstance(subst_tuple[0], DerivIndicator):
            original_iterator = subst_tuple[0].get_derivative_iterator(SYMB, '_gauss', use_transposed_gradients)
        elif isinstance(subst_tuple[0], str):
            original_iterator = get_flat_list_of_components(SYMB[subst_tuple[0]+'_gauss'])
        else:
            raise "Substitution tuples should contain only strings or DerivIndicators"
        if isinstance(subst_tuple[1], DerivIndicator):
            # new_iterator_aux = subst_tuple[1].get_derivative_iterator(SYMB, '_array', use_transposed_gradients) # REMOVE THIS
            new_iterator = subst_tuple[1].get_derivative_iterator_with_substitutions(SYMB, '_array', use_transposed_gradients, substitute_all_placeholders_gauss)
        elif isinstance(subst_tuple[1], str):
            new_iterator_aux = get_flat_list_of_components(SYMB[subst_tuple[1]+'_array'])
            new_iterator = [substitute_all_placeholders_gauss(new_elem_aux) for new_elem_aux in new_iterator_aux]
        else:
            raise "Substitution tuples should contain only strings or DerivIndicators"
        assert len(original_iterator)==len(new_iterator)
        for i in range(len(original_iterator)):
            lhs = lhs.subs(original_iterator[i],new_iterator[i])

    lhs = lhs.doit()
    print(str(lhs))

    lhs_out = OutputMatrix_CollectingFactors(w_g*lhs, "rLeftHandSideMatrix", mode, indentation_level=2, assignment_op="+=")
    
    full_output_string = f"Functional in continuous, compact form:\n$${latex_out_cont_compact}$$\n\nFunctional in continuous, compact form after substituting predefined functions:\n$${latex_out_cont_compact_2}$$\n\nFunctional in continuous, expanded form:\n$${latex_out_cont_expanded}$$\n\nFunctional in discretized from:\n$${latex_out_gauss}$$"
    # full_output_string = full_output_string.replace('array', '')
    # full_output_string = full_output_string.replace('gauss', '')
    # full_output_string = full_output_string.replace('\\left(x,y \\right)', '')

    

    cpp_interface_json = generate_interface_json(dim, nnodes, placeholders_list, sp.shape(dofs)[0])

    return rhs_out, lhs_out, full_output_string, cpp_interface_json