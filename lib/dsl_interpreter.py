import lib.placeholder_operators as po
from lib.basic_classes import DerivIndicator

class DSLInterpreter():
    
    def __init__(self, ssa, namespace):
        self.ssa = ssa
        self.namespace = namespace

        self.operator_map = {
            "add": self.add,
            "doublecontract": po.doublecontract_op,
            "grad": po.grad_op,
            "norm": po.norm_op,
            "symgrad": po.symgrad_op,
            "div": po.div_op,
            "curl": po.curl_op,
            "curl_2d": po.curl_2d_op,
            "mat_prod": po.matrix_prod_op,
            "transpose": po.matrix_transpose_op,
            "mat_vec_prod": po.matrix_vector_prod_op,
            "mat_det": po.matrix_det_op,
            "mat_inv": po.matrix_inv_op,
            "mat_cofactor": po.matrix_cofactor_op,
            "cross_prod": po.vector_cross_prod_op,
            "vec_outer_prod": po.vector_outer_prod_op
        }

    def interpret_single_abs(self,abs):
        if isinstance(abs["args"], list):
            for i, arg in enumerate(abs["args"]):
                if isinstance(arg, dict) and "op" in arg:
                    abs["args"][i] = self.interpret_single_abs(arg)
                elif isinstance(arg, str):
                    abs["args"][i] = self.namespace[arg]
                elif isinstance(arg, (int, float)):
                    abs["args"][i] = arg
                else:
                    raise TypeError(f"Unknown arg type: {type(arg)}")
        else:
            raise TypeError("args must be a list")
        return self.operator_map[abs["op"]](*abs["args"])
    
    def interpret_ssa(self):
        for abs in self.ssa:
            abs_copy = abs.copy() # avoid modifying original
            self.namespace[abs_copy["name"]] = self.interpret_single_abs(abs_copy["AST"])

    def add(self, *args):
        if len(args) == 0:
            raise ValueError("At least one argument required in add function")
        elif len(args) == 1:
            return args[0]
        elif len(args) == 2:
            return args[0] + args[1]
        else:
            return args[0] + self.add(*args[1:])
        
class SubstitutionsDSLInterpreter():

    def __init__(self, substitutions_dict):
        self.substitutions_dict = substitutions_dict

    def interpret_single_component(self,component):
        if component["op"] == "array":
            return component["args"][0]
        elif component["op"] == "deriv":
            return DerivIndicator(component["args"][0], component["args"][1])
        raise TypeError("Unrecognized operator")
    
    def interpret_substitutions(self):
        out_subst_dict = {}
        for subst_group_key, subst_group_val in self.substitutions_dict.items():
            out_subst_dict[subst_group_key] = [] 
            for subst_list in subst_group_val:
                out_subst_dict[subst_group_key].append(tuple([self.interpret_single_component(component) for component in subst_list]))
        return out_subst_dict