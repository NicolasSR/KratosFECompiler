import lib.placeholder_operators as po

class DSLInterpreter():
    
    def __init__(self, ssa, namespace):
        self.ssa = ssa
        self.namespace = namespace

        self.operator_map = {
            "add": self.add,
            "doublecontract_op": po.doublecontract_op,
            "grad_op": po.grad_op,
            "norm_op": po.norm_op
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
