import importlib

import sympy as sp
from sympy.printing.precedence import PRECEDENCE, PRECEDENCE_TRADITIONAL

from lib.basic_classes import BaseTensorPlaceholder, DeferredAdd, DeferredMul, check_if_scalar
from lib.printers import CustomLatexPrinter
from lib.registry import load_operators
from lib.yaml_utils import evaluate_yaml_operators_tree
from lib.coordinates_system import ACTIVE_COORD_SYSTEM

## Define tensorial operators

class DeferredTensorOp(sp.Expr):
    is_commutative = True  # This allows SymPy to automatically simplify expressions like a + b + c, even if a, b, c are DeferredTensorOps

    operators_config = load_operators() # Load operator definitions from YAML

    def __new__(cls, *args):
        # Validation: get list of ranks and dims and apply custom checks
        ranks, dims = cls._get_ranks_and_dims(args)
        config = cls.operators_config.get(cls.__name__, None)
        if config is None:
            raise ValueError(f"No operator configuration found for {cls.__name__}")
        arity = config.get('arity', None)
        if arity is None:
            raise ValueError(f"Operator {cls.__name__} has no arity defined")
        if len(args) != arity:
            raise ValueError(f"{cls.__name__} expects exactly {arity} arguments, got {len(args)}")
        constraints = config.get('constraints', {})
        if 'rank' in constraints.keys():
            is_valid, err_str = cls._validate_inputs(ranks, constraints, 'rank')
            if not is_valid:
                raise ValueError(err_str)
        if arity > 1 and 'dim' in constraints.keys():
            is_valid, err_str = cls._validate_inputs(dims, constraints, 'dim')
            if not is_valid:
                raise ValueError(err_str)
        return sp.Expr.__new__(cls, *args)
    
    @staticmethod
    def _get_ranks_and_dims(args):
        ranks = []
        dims = []
        for arg in args:
            if check_if_scalar(arg):
                rank = 0
                dim = []
            elif isinstance(arg, sp.NDimArray):
                dim_aux = list(arg.shape)
                if sum(dim_aux) == 1:
                    rank = 0
                    dim = []
                else:
                    rank = arg.rank()
                    dim = dim_aux
            else:
                rank = getattr(arg, 'rank', None)
                dim = getattr(arg, 'dim', None)
            ranks.append(rank)
            dims.append(dim)
        return ranks, dims
    
    @classmethod
    def _validate_inputs(cls, values_list, constraints, quantity_name):
        vars = {'a': values_list[0], 'b': values_list[1]} if len(values_list) == 2 else {'a': values_list[0]}
        is_valid = evaluate_yaml_operators_tree(constraints[quantity_name], vars)
        if not is_valid:
            return False, f"{quantity_name} constraints not satisfied in {cls.__name__}. Values: {values_list}"
        return True, ""

    @property
    def rank(self):
        ranks, _ = self._get_ranks_and_dims(self.args)
        vars = {'a': ranks[0], 'b': ranks[1]} if len(self.args) == 2 else {'a': ranks[0]}
        class_name = self.__class__.__name__
        output_config = self.operators_config[class_name]['output']
        return evaluate_yaml_operators_tree(output_config['rank'], vars)
    
    @property
    def dim(self):
        _, dims = self._get_ranks_and_dims(self.args)
        vars = {'a': dims[0], 'b': dims[1]} if len(self.args) == 2 else {'a': dims[0]}
        class_name = self.__class__.__name__
        output_config = self.operators_config[class_name]['output']
        return evaluate_yaml_operators_tree(output_config['dim'], vars)

    def evaluate(self, *args):
        # Resolve dependencies recursively
        resolved_args = [arg.evaluate() if hasattr(arg, 'evaluate') else arg for arg in self.args]
        resolved_args.extend([arg.evaluate() if hasattr(arg, 'evaluate') else arg for arg in args])

        # Perform the actual numeric math
        if all(not arg.has(BaseTensorPlaceholder) for arg in resolved_args):
            class_name = self.__class__.__name__
            implementation_name = self.operators_config[class_name]['implementation']
            
            # Dynamically loads the function from the tensor operators file.
            try:
                # Import the module
                module = importlib.import_module('lib.tensor_operators')
                # Get the function object by name
                op_func = getattr(module, implementation_name)
                # Return the result of the function call with the resolved arguments
                return op_func(*resolved_args)
            except (ImportError, AttributeError) as e:
                raise ValueError(f"Could not find implementation '{implementation_name}': {e}")
        
        # If still symbolic, return the symbolic representation
        return self.func(*resolved_args)
    
    # Algebraic operators

    def __add__(self, other):
        # return add_op(self, other)
        return DeferredAdd(self, other)
    
    def __radd__(self, other):
        # return add_op(other, self)
        return DeferredAdd(other, self)
    
    def __mul__(self, other):
        # return prod_op(self, other)
        return DeferredMul(self, other)
    
    def __rmul__(self, other):
        # return prod_op(other, self)
        return DeferredMul(other, self)
    
    # Printing
    def _latex(self, printer, exp=None, *args):
        if isinstance(printer, CustomLatexPrinter):
            class_name = self.__class__.__name__
            latex_pattern = self.operators_config[class_name]['latex']
            a = printer._print(self.args[0])
            b = printer._print(self.args[1]) if len(self.args) > 1 else None
            latex_string = latex_pattern.format(a=a, b=b)
            if exp is None:
                return latex_string
            else:
                if self.operators_config[class_name].get('needs_parentheses_when_exp', True):
                    latex_string = r"\left(%s\right)" % latex_string
                return r"{%s}^{%s}" % (latex_string, exp)
        else:
            return printer._print_Function(self, exp=exp)
    
    def __format__(self, format_spec: str):
        # This forces f-strings to use your __str__ method, 
        # and then apply standard string formatting on top of that.
        return sp.sstr(self).__format__(format_spec)
    

class norm_op(DeferredTensorOp):
    # It inherits all behavior from DeferredTensorOp 
    # which uses the YAML definition for 'norm_op'.
    pass


class dot_op(DeferredTensorOp):
    # It inherits all behavior from DeferredTensorOp 
    # which uses the YAML definition for 'dot_op'.
    pass
    
        
class contract_op(DeferredTensorOp):
    # It inherits all behavior from DeferredTensorOp 
    # which uses the YAML definition for 'contract_op'.
    pass

class doublecontract_op(DeferredTensorOp):
    # It inherits all behavior from DeferredTensorOp 
    # which uses the YAML definition for 'doublecontract_op'.
    pass
        
class grad_op(DeferredTensorOp): # Returns dfj/dxi
    
    def evaluate(self):
        # Retrieve the current active system
        coords = ACTIVE_COORD_SYSTEM.get()["coord_symbols"]
        if coords is None:
            raise RuntimeError("grad_op called outside of a 'with CoordinateSystem(...)' block.")
            
        # Add coordinate symbols list as an argument to the parent's evaluate() mehtod
        return super().evaluate(coords)
    
    @property
    def dim(self):
        _, dims = self._get_ranks_and_dims(self.args)
        coord_dims = [len(ACTIVE_COORD_SYSTEM.get()["coord_symbols"])]
        vars = {'a': dims[0], 'b': dims[1]} if len(self.args) == 2 else {'a': dims[0], 'coords': coord_dims}
        class_name = self.__class__.__name__
        output_config = self.operators_config[class_name]['output']
        return evaluate_yaml_operators_tree(output_config['dim'], vars)

# class grad_transposed_op(sp.Function):  # Returns dfi/dxj
#     precedence = PRECEDENCE_TRADITIONAL['Gradient']

#     @classmethod
#     def eval(cls, a, base_scalars):
#         if not a.has(BaseTensorPlaceholder) and not base_scalars.has(BaseTensorPlaceholder):
#             return tensor_transpose_grad(a,base_scalars)
        
#     def _latex(self, printer, exp=None, *args):
#         if isinstance(printer, CustomLatexPrinter):
#             a, base_scalars = self.args
#             rv = r"\nabla %s" % printer.parenthesize(a, PRECEDENCE['Mul'])
#             if exp is None:
#                 return rv
#             else:
#                 return r"\left(%s\right)^{%s}" % (rv, exp)
#         else:
#             return printer._print_Function(self, exp=exp)
        
class symgrad_op(sp.Function):
    precedence = PRECEDENCE_TRADITIONAL['Gradient']

    @classmethod
    def eval(cls, a, base_scalars):
        if not a.has(BaseTensorPlaceholder) and not base_scalars.has(BaseTensorPlaceholder):
            return tensor_symgrad(a,base_scalars)
        
    def _latex(self, printer, exp=None, *args):
        if isinstance(printer, CustomLatexPrinter):
            a, base_scalars = self.args
            rv = r"\nabla^{s} %s" % printer.parenthesize(a, PRECEDENCE['Mul'])
            if exp is None:
                return rv
            else:
                return r"\left(%s\right)^{%s}" % (rv, exp)
        else:
            return printer._print_Function(self, exp=exp)
        
class div_op(sp.Function):
    precedence = PRECEDENCE_TRADITIONAL['Divergence']

    @classmethod
    def eval(cls, a, base_scalars):
        if not a.has(BaseTensorPlaceholder) and not base_scalars.has(BaseTensorPlaceholder):
            return tensor_div(a,base_scalars)
        
    def _latex(self, printer, exp=None, *args):
        if isinstance(printer, CustomLatexPrinter):
            a, base_scalars = self.args
            rv = r"\nabla\cdot %s" % printer.parenthesize(a, PRECEDENCE['Mul'])
            if exp is None:
                return rv
            else:
                return r"\left(%s\right)^{%s}" % (rv, exp)
        else:
            return printer._print_Function(self, exp=exp)
        
# Define matrix operations:

class matrix_prod_op(sp.Function):
    precedence = PRECEDENCE_TRADITIONAL['Dot'] # Probably wrong

    @classmethod
    def eval(cls, a, b):
        if not a.has(BaseTensorPlaceholder) and not b.has(BaseTensorPlaceholder):
            return tensor_mat_prod(a,b)
        
    def _latex(self, printer, exp=None, *args):
        if isinstance(printer, CustomLatexPrinter):
            a, b = self.args
            rv = r"%s %s" % (printer.parenthesize(a, PRECEDENCE['Mul']),
                                 printer.parenthesize(b, PRECEDENCE['Mul']))
            if exp is None:
                return rv
            else:
                return r"\left(%s\right)^{%s}" % (rv, exp)
        else:
            return printer._print_Function(self, exp=exp)
        
class matrix_transpose_op(sp.Function):
    precedence = PRECEDENCE_TRADITIONAL['Dot'] # Probably wrong

    @classmethod
    def eval(cls, a):
        if not a.has(BaseTensorPlaceholder):
            return tensor_mat_transpose(a)
        
    def _latex(self, printer, exp=None, *args):
        if isinstance(printer, CustomLatexPrinter):
            a = self.args[0]
            rv = r"%s^T" % printer.parenthesize(a, PRECEDENCE['Mul'])
            if exp is None:
                return rv
            else:
                return r"\left(%s\right)^{%s}" % (rv, exp)
        else:
            return printer._print_Function(self, exp=exp)
        
class matrix_vector_prod_op(sp.Function):
    # Specifically return b = Ax
    # For b^T = x^T A, the user may just use a single contract operation
    precedence = PRECEDENCE_TRADITIONAL['Dot'] # Probably wrong

    @classmethod
    def eval(cls, a, b):
        if not a.has(BaseTensorPlaceholder) and not b.has(BaseTensorPlaceholder):
            return tensor_mat_vector_prod(a,b)
        
    def _latex(self, printer, exp=None, *args):
        if isinstance(printer, CustomLatexPrinter):
            a, b = self.args
            rv = r"%s %s" % (printer.parenthesize(a, PRECEDENCE['Mul']),
                                 printer.parenthesize(b, PRECEDENCE['Mul']))
            if exp is None:
                return rv
            else:
                return r"\left(%s\right)^{%s}" % (rv, exp)
        else:
            return printer._print_Function(self, exp=exp)
        
class matrix_det_op(sp.Function):
    precedence = PRECEDENCE_TRADITIONAL['Dot'] # Probably wrong

    @classmethod
    def eval(cls, a):
        if not a.has(BaseTensorPlaceholder):
            return tensor_mat_det(a)
        
    def _latex(self, printer, exp=None, *args):
        if isinstance(printer, CustomLatexPrinter):
            a = self.args[0]
            rv = r"|%s|" % printer.doprint(a)
            if exp is None:
                return rv
            else:
                return r"%s^{%s}" % (rv, exp)
        else:
            return printer._print_Function(self, exp=exp)
        
class matrix_inv_op(sp.Function):
    precedence = PRECEDENCE_TRADITIONAL['Dot'] # Probably wrong

    @classmethod
    def eval(cls, a):
        if not a.has(BaseTensorPlaceholder):
            return tensor_mat_inv(a)
        
    def _latex(self, printer, exp=None, *args):
        if isinstance(printer, CustomLatexPrinter):
            a = self.args[0]
            rv = r"%s^{-1}" % printer.parenthesize(a, PRECEDENCE['Mul'])
            if exp is None:
                return rv
            else:
                return r"%s^{%s}" % (rv, exp)
        else:
            return printer._print_Function(self, exp=exp)
        
class matrix_cofactor_op(sp.Function):
    precedence = PRECEDENCE_TRADITIONAL['Dot'] # Probably wrong

    @classmethod
    def eval(cls, a):
        if not a.has(BaseTensorPlaceholder):
            return tensor_mat_cofactor(a)
        
    def _latex(self, printer, exp=None, *args):
        if isinstance(printer, CustomLatexPrinter):
            a = self.args[0]
            if exp is None:
                return r"\text{Cof}%s" % printer.parenthesize(a, PRECEDENCE['Mul'])
            else:
                return r"\text{Cof}^{%s}%s" % (exp,printer.parenthesize(a, PRECEDENCE['Mul']))
        else:
            return printer._print_Function(self, exp=exp)

class vector_cross_prod_op(sp.Function):
    precedence = PRECEDENCE_TRADITIONAL['Dot'] # Probably wrong

    @classmethod
    def eval(cls, a, b):
        if not a.has(BaseTensorPlaceholder) and not b.has(BaseTensorPlaceholder):
            return tensor_vec_cross_prod(a,b)
        
    def _latex(self, printer, exp=None, *args):
        if isinstance(printer, CustomLatexPrinter):
            a, b = self.args
            rv = r"%s \times %s" % (printer.parenthesize(a, PRECEDENCE['Mul']),
                                 printer.parenthesize(b, PRECEDENCE['Mul']))
            if exp is None:
                return rv
            else:
                return r"\left(%s\right)^{%s}" % (rv, exp)
        else:
            return printer._print_Function(self, exp=exp)

class vector_outer_prod_op(sp.Function):
    precedence = PRECEDENCE_TRADITIONAL['Dot'] # Probably wrong

    @classmethod
    def eval(cls, a, b):
        if not a.has(BaseTensorPlaceholder) and not b.has(BaseTensorPlaceholder):
            return tensor_vec_outer_prod(a,b)
        
    def _latex(self, printer, exp=None, *args):
        if isinstance(printer, CustomLatexPrinter):
            a, b = self.args
            rv = r"%s \otimes %s" % (printer.parenthesize(a, PRECEDENCE['Mul']),
                                 printer.parenthesize(b, PRECEDENCE['Mul']))
            if exp is None:
                return rv
            else:
                return r"\left(%s\right)^{%s}" % (rv, exp)
        else:
            return printer._print_Function(self, exp=exp)

class flatten_and_combine_tensors(sp.Function):
    precedence = PRECEDENCE_TRADITIONAL['Dot'] # Probably wrong

    @classmethod
    def eval(cls, a):
        if not a.has(BaseTensorPlaceholder):
            return tensor_arrays_flatten_and_combine(a)
        
    def _latex(self, printer, exp=None, *args):
        if isinstance(printer, CustomLatexPrinter):
            a = self.args[0]
            rv = r"%s" % printer.doprint(a)
            if exp is None:
                return rv
            else:
                return r"%s^{%s}" % (rv, exp)
        else:
            return printer._print_Function(self, exp=exp)