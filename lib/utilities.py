import re

# Probably should be in another utilities file and given a more specific name
def substitute_symbols(input_string):
    # This pattern matches anything between */* and */*
    pattern = r'¨(.*?)¨'
    # This substitution replaces the match with 'var[X]' where X is the captured group
    return re.sub(pattern, r"SYMB['\1']", input_string)

# Probably should be in another utilities file and given a more specific name
def substitute_functions(input_string):
    substitutions_list = [
        ('add_op(', 'add_op('),
        ('prod_op(', 'prod_op('),
        ('sub_op(', 'sub_op('),
        ('symgrad(', 'po.symgrad_op('),
        ('grad(', 'po.grad_op('),
        ('div(', 'po.div_op('),
        ('norm(', 'po.norm_op('),
        ('det(', 'po.matrix_det_op('),
        ('cofactor(', 'po.matrix_cofactor_op('),
        ('double_contract(', 'po.doublecontract_op('),
        ('contract(', 'po.contract_op('),
        ('matrix_prod(', 'po.matrix_prod_op('),
        ('transpose(', 'po.matrix_transpose_op('),
        ('dot(', 'po.dot_op('),
        ('matrix_vector_prod(', 'po.matrix_vector_prod_op(')
    ]
    for old, new in substitutions_list:
        input_string = input_string.replace(old, new)
    return input_string

def substitute_all_placeholders_in_expression(expr, placeholder_names_list, namespace):
    subst_list = [( namespace[name],  namespace[name].array) for name in placeholder_names_list] 
    expr = expr.subs(subst_list, evaluate=False)
    return expr

def substitute_all_arrays_in_expression(expr, placeholder_names_list, namespace):
    for placeholder_name in reversed(placeholder_names_list):
        placeholder = namespace[placeholder_name]
        expr = placeholder.substitute_arrays_to_gauss(expr)
    return expr