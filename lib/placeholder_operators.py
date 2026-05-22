import importlib

import sympy as sp
from sympy.printing.precedence import PRECEDENCE, PRECEDENCE_TRADITIONAL

from lib.basic_classes import BaseTensorPlaceholder, DeferredAdd, DeferredMul, check_if_scalar
from lib.printers import CustomLatexPrinter
from lib.registry import load_operators
from lib.yaml_utils import evaluate_yaml_operators_tree
from lib.coordinates_system import ACTIVE_COORD_SYSTEM

OPERATORS_CONFIG = load_operators() # Load operator definitions from YAML

## Define tensorial operators

class DeferredTensorOp(sp.Expr):
    is_commutative = True  # This allows SymPy to automatically simplify expressions like a + b + c, even if a, b, c are DeferredTensorOps

    def __new__(cls, *args):
        config = OPERATORS_CONFIG.get(cls.__name__, None)
        if config is None:
            raise ValueError(f"No operator configuration found for {cls.__name__}")
        
        # Retrieve the current active system
        coords = ACTIVE_COORD_SYSTEM.get()["coord_symbols"]
        transposed_gradients_flag = ACTIVE_COORD_SYSTEM.get()["transposed_gradients_flag"]
        if coords is None:
            raise RuntimeError("Operator called outside of a 'with CoordinateSystem(...)' block.")

        arity = config.get('arity', None)
        if arity is None:
            raise ValueError(f"Operator {cls.__name__} has no arity defined")
        if len(args) != arity:
            raise ValueError(f"{cls.__name__} expects exactly {arity} arguments, got {len(args)}")
        

        # Validation: get list of ranks and dims and apply custom checks
        pass_coords = config.get('pass_coords', False)
        if pass_coords:
            ranks, dims = cls._get_ranks_and_dims(args + (coords,))
        else:
            ranks, dims = cls._get_ranks_and_dims(args)
        
        constraints = cls._get_field_from_correct_version(config, 'constraints', {}, transposed_gradients_flag)
        if 'rank' in constraints.keys():
            is_valid, err_str = cls._validate_inputs(ranks, constraints, 'rank', pass_coords)
            if not is_valid:
                raise ValueError(err_str)
        if 'dim' in constraints.keys():
            is_valid, err_str = cls._validate_inputs(dims, constraints, 'dim', pass_coords)
            if not is_valid:
                raise ValueError(err_str)
        return sp.Expr.__new__(cls, *args)
    
    @staticmethod
    def _get_field_from_correct_version(config, field_name, default_val, transposed_gradients_flag):
        # Apply alternate versions of the operator if specified in the YAML config based on the flags, ohterwise original config
        field = config.get(field_name, default_val)
        if "alternate_version" in config.keys():
            for version_name in config["alternate_version"].keys():
                if version_name == 'transposed_gradients_flag' and transposed_gradients_flag:
                    field = config["alternate_version"]["transposed_gradients_flag"].get(field_name, field)
        return field
    
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
    def _validate_inputs(cls, values_list, constraints, quantity_name, pass_coords):
        if pass_coords:
            vars = {'a': values_list[0], 'b': values_list[1], 'coords': values_list[2]} if len(values_list) == 3 else {'a': values_list[0], 'coords': values_list[1]}
        else:
            vars = {'a': values_list[0], 'b': values_list[1]} if len(values_list) == 2 else {'a': values_list[0]}
        is_valid = evaluate_yaml_operators_tree(constraints[quantity_name], vars)
        if not is_valid:
            return False, f"{quantity_name} constraints not satisfied in {cls.__name__}. Values: {values_list}"
        return True, ""

    @property
    def rank(self):
        class_name = self.__class__.__name__
        config = OPERATORS_CONFIG[class_name]
        coords = ACTIVE_COORD_SYSTEM.get()["coord_symbols"]
        pass_coords = config.get('pass_coords', False)

        if pass_coords:
            ranks, _ = self._get_ranks_and_dims(self.args+(coords,))
            vars = {'a': ranks[0], 'b': ranks[1], 'coords': ranks[2]} if len(ranks) == 3 else {'a': ranks[0], 'coords': ranks[1]}
        else:
            ranks, _ = self._get_ranks_and_dims(self.args)
            vars = {'a': ranks[0], 'b': ranks[1]} if len(ranks) == 2 else {'a': ranks[0]}

        transposed_gradients_flag = ACTIVE_COORD_SYSTEM.get()["transposed_gradients_flag"]
        output_config = self._get_field_from_correct_version(config, 'output', {}, transposed_gradients_flag)

        return evaluate_yaml_operators_tree(output_config['rank'], vars)
    
    @property
    def dim(self):
        class_name = self.__class__.__name__
        config = OPERATORS_CONFIG[class_name]
        coords = ACTIVE_COORD_SYSTEM.get()["coord_symbols"]
        pass_coords = config.get('pass_coords', False)

        if pass_coords:
            _, dims = self._get_ranks_and_dims(self.args+(coords,))
            vars = {'a': dims[0], 'b': dims[1], 'coords': dims[2]} if len(dims) == 3 else {'a': dims[0], 'coords': dims[1]}
        else:
            _, dims = self._get_ranks_and_dims(self.args)
            vars = {'a': dims[0], 'b': dims[1]} if len(dims) == 2 else {'a': dims[0]}

        transposed_gradients_flag = ACTIVE_COORD_SYSTEM.get()["transposed_gradients_flag"]
        output_config = self._get_field_from_correct_version(config, 'output', {}, transposed_gradients_flag)
        return evaluate_yaml_operators_tree(output_config['dim'], vars)

    def evaluate(self):
        class_name = self.__class__.__name__
        config = OPERATORS_CONFIG[class_name]
        transposed_gradients_flag = ACTIVE_COORD_SYSTEM.get()["transposed_gradients_flag"]
        coords = ACTIVE_COORD_SYSTEM.get()["coord_symbols"]

        # Resolve dependencies recursively
        resolved_args = [arg.evaluate() if hasattr(arg, 'evaluate') else arg for arg in self.args]
        pass_coords = config.get('pass_coords', False)
        if pass_coords:
            resolved_args.append(coords.evaluate() if hasattr(coords, 'evaluate') else coords)

        # Perform the actual tensor math
        if all(not arg.has(BaseTensorPlaceholder) for arg in resolved_args):
            implementation_name = self._get_field_from_correct_version(config, 'implementation', None, transposed_gradients_flag)
            
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
            config = OPERATORS_CONFIG[class_name]
            a = printer._print(self.args[0])
            b = printer._print(self.args[1]) if len(self.args) > 1 else None
            latex_pattern = config.get('latex', None)
            latex_string = latex_pattern.format(a=a, b=b)
            if exp is None:
                return latex_string
            else:
                if config.get('needs_parentheses_when_exp', True):
                    latex_string = r"\\left(%s\\right)" % latex_string
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
        
class grad_op(DeferredTensorOp):
    # It inherits all behavior from DeferredTensorOp 
    # which uses the YAML definition for 'doublecontract_op'.
    pass

class symgrad_op(DeferredTensorOp):
    # It inherits all behavior from DeferredTensorOp 
    # which uses the YAML definition for 'doublecontract_op'.
    pass

class div_op(DeferredTensorOp):
    # It inherits all behavior from DeferredTensorOp 
    # which uses the YAML definition for 'div_op'.
    pass

class curl_op(DeferredTensorOp):
    # It inherits all behavior from DeferredTensorOp
    # which uses the YAML definition for 'curl_op'.
    pass
        
class curl_2d_op(DeferredTensorOp):
    # It inherits all behavior from DeferredTensorOp
    # which uses the YAML definition for 'curl_2d_op'.
    pass

class matrix_prod_op(DeferredTensorOp):
    # It inherits all behavior from DeferredTensorOp 
    # which uses the YAML definition for 'matrix_prod_op'.
    pass
        
class matrix_transpose_op(DeferredTensorOp):
    # It inherits all behavior from DeferredTensorOp 
    # which uses the YAML definition for 'matrix_transpose_op'.
    pass

class matrix_vector_prod_op(DeferredTensorOp):
    # It inherits all behavior from DeferredTensorOp 
    # which uses the YAML definition for 'matrix_vector_prod_op'.
    pass
        
class matrix_det_op(DeferredTensorOp):
    # It inherits all behavior from DeferredTensorOp 
    # which uses the YAML definition for 'matrix_det_op'.
    pass
        
class matrix_inv_op(DeferredTensorOp):
    # It inherits all behavior from DeferredTensorOp 
    # which uses the YAML definition for 'matrix_inv_op'.
    pass
        
class matrix_cofactor_op(DeferredTensorOp):
    # It inherits all behavior from DeferredTensorOp 
    # which uses the YAML definition for 'matrix_cofactor_op'.
    pass

class vector_cross_prod_op(DeferredTensorOp):
    # It inherits all behavior from DeferredTensorOp 
    # which uses the YAML definition for 'vector_cross_prod_op'.
    pass

class vector_outer_prod_op(DeferredTensorOp):
    # It inherits all behavior from DeferredTensorOp 
    # which uses the YAML definition for 'vector_outer_prod_op'.
    pass

# class flatten_and_combine_tensors(sp.Function):
#     precedence = PRECEDENCE_TRADITIONAL['Dot'] # Probably wrong

#     @classmethod
#     def eval(cls, a):
#         if not a.has(BaseTensorPlaceholder):
#             return tensor_arrays_flatten_and_combine(a)
        
#     def _latex(self, printer, exp=None, *args):
#         if isinstance(printer, CustomLatexPrinter):
#             a = self.args[0]
#             rv = r"%s" % printer.doprint(a)
#             if exp is None:
#                 return rv
#             else:
#                 return r"%s^{%s}" % (rv, exp)
#         else:
#             return printer._print_Function(self, exp=exp)