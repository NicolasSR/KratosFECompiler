import sympy as sp

from lib.components_utils import get_flat_list_of_components
from lib.coordinates_system import ACTIVE_COORD_SYSTEM


def substitute_all_placeholders_in_expression(expr, placeholder_names_list, namespace):
    subst_list = [( namespace[name],  namespace[name].array) for name in placeholder_names_list if expr.has(namespace[name])] 
    expr = expr.subs(subst_list, evaluate=False)
    return expr

def substitute_all_arrays_in_expression(expr, placeholder_names_list, namespace):
    for placeholder_name in reversed(placeholder_names_list):
        placeholder = namespace[placeholder_name]
        expr = placeholder.substitute_arrays_to_gauss(expr)
    return expr

def substitute_defined_functions_in_expression(expr, defined_functions_list, namespace):
    # Substitute defined functions by their definition (order matters!)
    for defined_function_name in defined_functions_list:
        defined_function = namespace[defined_function_name]
        function_definition = namespace[defined_function_name+"_def"]
        expr = expr.subs(defined_function, function_definition)
    return expr

def get_substitution_iterator_pre_lhs(component, placeholder_names_list, defined_functions_list, namespace):
    if isinstance(component, DerivIndicator):
        iterator = component.get_derivative_iterator_pre_lhs(placeholder_names_list, namespace)
    else:
        auxiliary_expr = substitute_defined_functions_in_expression(component, defined_functions_list, namespace)
        auxiliary_expr = substitute_all_placeholders_in_expression(auxiliary_expr, placeholder_names_list, namespace)
        if hasattr(auxiliary_expr, "evaluate") and callable(getattr(auxiliary_expr, "evaluate")):
            auxiliary_expr = auxiliary_expr.evaluate()
        iterator = get_flat_list_of_components(auxiliary_expr)
    return iterator

def get_substitution_iterator_post_lhs(component, placeholder_names_list, defined_functions_list, namespace):
    if isinstance(component, DerivIndicator):
        iterator = component.get_derivative_iterator_post_lhs(placeholder_names_list, defined_functions_list, namespace)
    else:
        expression = substitute_defined_functions_in_expression(component, defined_functions_list, namespace)
        expression = substitute_all_placeholders_in_expression(expression, placeholder_names_list, namespace)
        if hasattr(expression, "evaluate") and callable(getattr(expression, "evaluate")):
            expression = expression.evaluate()
        expression = substitute_all_arrays_in_expression(expression, placeholder_names_list, namespace)
        iterator = get_flat_list_of_components(expression)
    return iterator

class DerivIndicator():
    def __init__(self, function, var):
        self.function = function
        self.var = var
    
    def get_derivative_iterator_pre_lhs(self, placeholder_names_list, namespace):
        raise NotImplementedError("DerivIndicator not implemented for pre-LHS substitution")
    
    def get_derivative_iterator_post_lhs(self, placeholder_names_list, defined_functions_list, namespace):
        numerator_expression = substitute_defined_functions_in_expression(self.function, defined_functions_list, namespace)
        numerator_expression = substitute_all_placeholders_in_expression(numerator_expression, placeholder_names_list, namespace)
        if not (isinstance(numerator_expression, sp.NDimArray) or isinstance(numerator_expression, sp.Function)):
            numerator_expression = numerator_expression.evaluate()
        numerator_expression = substitute_all_arrays_in_expression(numerator_expression, placeholder_names_list, namespace)
        numerator_iterator = get_flat_list_of_components(numerator_expression)
        denominator_expression = substitute_defined_functions_in_expression(self.var, defined_functions_list, namespace)
        denominator_expression = substitute_all_placeholders_in_expression(denominator_expression, placeholder_names_list, namespace)
        if not (isinstance(denominator_expression, sp.NDimArray) or isinstance(denominator_expression, sp.Function)):
            denominator_expression = denominator_expression.evaluate()
        denominator_expression = substitute_all_arrays_in_expression(denominator_expression, placeholder_names_list, namespace)
        denominator_iterator = get_flat_list_of_components(denominator_expression)
        derivatives_list = []
        if ACTIVE_COORD_SYSTEM.get()["transposed_gradients_flag"]:
            for numerator in numerator_iterator:
                for denominator in denominator_iterator:
                    derivatives_list.append(sp.Derivative(numerator,denominator).doit())
        else:
            for denominator in denominator_iterator:
                for numerator in numerator_iterator:
                    derivatives_list.append(sp.Derivative(numerator,denominator).doit())
        return derivatives_list
    
    def __repr__(self):
        return "DerivIndicator("+self.function.__repr__()+","+self.var.__repr__()+")"