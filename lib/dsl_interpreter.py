import sympy as sp

import lib.placeholder_operators as po
from lib.substitution_utils import DerivIndicator

class DSLInterpreter():
    
    def __init__(self, namespace, defined_functions_list = [], config_settings_dict = dict()):
        self.namespace = namespace
        self.defined_functions_list = defined_functions_list
        self.config_settings_dict = config_settings_dict

        self.operator_map = {
            "add": self.add,
            "subtract": self.subtract,
            "mult": self.mult,
            "divide": self.divide,
            "unary_minus": self.unary_minus,
            "pow": self.pow,
            "dot_prod": po.dot_op,
            "contract":po.contract_op,
            "double_contract": po.doublecontract_op,
            "grad": po.grad_op,
            "norm": po.norm_op,
            "symgrad": po.symgrad_op,
            "div": po.div_op,
            "curl": po.curl_op,
            "curl_2d": po.curl_2d_op,
            "matrix_prod": po.matrix_prod_op,
            "matrix_transpose": po.matrix_transpose_op,
            "matrix_vector_prod": po.matrix_vector_prod_op,
            "matrix_determinant": po.matrix_det_op,
            "matrix_inv": po.matrix_inv_op,
            "matrix_cofactor": po.matrix_cofactor_op,
            "cross_prod": po.vector_cross_prod_op,
            "vec_outer_prod": po.vector_outer_prod_op,
            "if": self.if_statement
        }

    def interpret_single_abs(self,abs):
        print("Interpreting single abs: ", abs)
        if isinstance(abs["args"], list):
            for i, arg in enumerate(abs["args"]):
                if isinstance(arg, dict) and "op" in arg:
                    abs["args"][i] = self.interpret_single_abs(arg)
                elif isinstance(arg, str):
                    if arg in self.namespace.keys():
                        abs["args"][i] = self.namespace[arg]
                    elif arg in self.config_settings_dict.keys():
                        abs["args"][i] = self.config_settings_dict[arg]
                    else:
                        raise KeyError(f"Key {arg} not found in namespace or config settings")
                elif isinstance(arg, (int, float)):
                    abs["args"][i] = sp.sympify(arg)
                else:
                    raise TypeError(f"Unknown arg type: {type(arg)}")
        else:
            raise TypeError("args must be a list")
        return self.operator_map[abs["op"]](*abs["args"])
    
    def interpret_ssa(self, ssa):
        for abs in ssa:
            abs_copy = abs.copy() # avoid modifying original
            new_var_name = abs_copy["name"]
            if new_var_name in self.defined_functions_list:
                new_var_name += "_def" # avoid overwriting the defined function
            self.namespace[new_var_name] = self.interpret_single_abs(abs_copy["AST"])

    def add(self, *args):
        if len(args) == 0:
            raise ValueError("At least one argument required in add function")
        elif len(args) == 1:
            return args[0]
        elif len(args) == 2:
            return args[0] + args[1]
        else:
            return args[0] + self.add(*args[1:])
        
    def mult(self, *args):
        if len(args) == 0:
            raise ValueError("At least one argument required in multiply function")
        elif len(args) == 1:
            return args[0]
        elif len(args) == 2:
            return args[0] * args[1]
        else:
            return args[0] * self.mult(*args[1:])
    
    def unary_minus(self, *args):
        if len(args) == 1:
            return -args[0]
        else:
            raise ValueError("One argument required in unary_minus function") 
        
    def subtract(self, *args):
        if len(args) == 2:
            return args[0] + self.unary_minus(args[1]) # args[0] - args[1]
        else:
            raise ValueError("Two arguments required in subtract function")
        
    def divide(self, *args):
        if len(args) == 2:
            return args[0] / args[1]
        else:
            raise ValueError("Two arguments required in division function")
        
    def pow(self, *args):
        if len(args) == 2:
            return args[0] ** args[1]
        else:
            raise ValueError("Two arguments required in pow function")
        
    def if_statement(self, *args):
        if len(args) == 3:
            condition = args[0]
            true_val = args[1]
            false_val = args[2]
            return true_val if condition else false_val
        else:
            raise ValueError("Three arguments required in If statement")
        
        
class SubstitutionsDSLInterpreter():

    def __init__(self, substitutions_dict, namespace):
        self.substitutions_dict = substitutions_dict
        self.namespace = namespace

    def interpret_single_component(self,component):
        if isinstance(component,str):
            return self.namespace[component]
        elif component["op"] == "deriv":
            args = [self.namespace[arg] for arg in component["args"]]
            return DerivIndicator(*args)
        raise TypeError("Unrecognized operator")
        # Maybe there should be an option for CoefficentsIndicator, CoordsIndicator and VarsCombination.
        # For now, x, y, z, base_scalars and dofs are in namespace, so the first case is enough.
    
    def interpret_substitutions(self):
        out_subst_dict = {}
        for subst_group_key, subst_group_val in self.substitutions_dict.items():
            out_subst_dict[subst_group_key] = [] 
            for subst_list in subst_group_val:
                out_subst_dict[subst_group_key].append(tuple([self.interpret_single_component(component) for component in subst_list]))
        return out_subst_dict