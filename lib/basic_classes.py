from functools import reduce
import operator
import importlib

import sympy as sp
from sympy import NDimArray

from lib.components_utils import get_flat_list_of_components
from lib.printers import CustomLatexPrinter
from lib.yaml_utils import evaluate_yaml_operators_tree

from lib.coordinates_system import ACTIVE_COORD_SYSTEM
from lib.registry import load_operators

OPERATORS_CONFIG = load_operators() # Load operator definitions from YAML

def check_if_scalar(arg):
    return  str(sp.sympify(arg).kind) == "NumberKind" or isinstance(arg, sp.Function)

def check_if_placeholder_or_deferred(arg):
    return isinstance(arg, (BaseTensorPlaceholder, DeferredArithmetic, DeferredTensorOp))

class CompilerNamespace(dict):
    def __getitem__(self, key):
        # Intercept base_scalars key to dynamically return the value from ACTIVE_COORD_SYSTEM
        if key == "base_scalars":
            if ACTIVE_COORD_SYSTEM.get() is None:
                raise Exception("No active coordinate system.")
            else:
                return ACTIVE_COORD_SYSTEM.get()["coord_symbols"]
        
        # Fall back to normal dictionary behavior for everything else
        return super().__getitem__(key)

class DeferredArithmetic(sp.Expr):
    """
    Common base class for DeferredAdd and DeferredMul, to handle arithmetics and printing.
    """

    _op_priority = 100 # Give a vey high priority so deferred arithmetic operations are used whnever a  deferred operation is encountered (in left or right)

    @staticmethod
    def _get_ranks_list(args):
        ranks = []
        for arg in args:
            if check_if_scalar(arg):
                ranks.append(0)
            elif isinstance(arg, NDimArray):
                dim_aux = list(arg.shape)
                if sum(dim_aux) == 1:
                    ranks.append(0)
                else:
                    ranks.append(arg.rank())
            else:
                ranks.append(getattr(arg, 'rank', None))
        return ranks

    # Algebraic operators

    def __add__(self, other):
        # return add_op(self, other)
        return DeferredAdd(self, other)
    
    def __radd__(self, other):
        # return add_op(other, self)
        return DeferredAdd(other, self)
    
    def __neg__(self):
        return DeferredNegative(self)
    
    def __mul__(self, other):
        # return prod_op(self, other)
        return DeferredMul(self, other)
    
    def __rmul__(self, other):
        # return prod_op(other, self)
        return DeferredMul(other, self)
    
    def __truediv__(self, other):
        return DeferredDivide(self, other)
    
    def __rtruediv__(self, other):
        return DeferredDivide(other, self)
    
    def __pow__(self, other):
        return DeferredPow(self, other)
    
    def __rpow__(self, other):
        return DeferredPow(other, self)
    
    # Make it into an iterator (for correct behaviour of iterargs in .has(), etc.)
    def __len__(self):
        return len(self.args)
    
    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count < len(self):
            out = self.args[self.count]
            self.count += 1
            return out
        raise StopIteration
    
    def _evaluate(self, fun_array, class_scalar):
        # Recursively evaluate children first
        evaluated_args = [arg.evaluate() if hasattr(arg, 'evaluate') else arg 
                          for arg in self.args]
        
        if any([check_if_placeholder_or_deferred(evaluated_arg) for evaluated_arg in evaluated_args]):
            return self
        elif any([isinstance(evaluated_arg, NDimArray) for evaluated_arg in evaluated_args]):
            return fun_array(*evaluated_args) # Let SymPy handle array power
        else:
            return class_scalar(*evaluated_args) # Fallback to standard SymPy for scalars
    
    # Printing
    
    def _sympystr(self, printer, exp=None):
        # Print each argument. doprint() applies recursion.
        args_str = [printer.doprint(arg) for arg in self.args]
        
        # Join them with the adequate operator
        # Parentehsis are added to prevent ambiguity
        result = f"({self.operator.join(args_str)})"
        if exp is not None:
            result = f"{result}**{printer.doprint(exp)}"
        return result

    def _latex(self, printer, exp=None):
        # Print each argument via the LaTeX printer. doprint() applies recursion.
        args_latex = [printer.doprint(arg) for arg in self.args]
        
        # Join using the LaTeX-specific operator (e.g., r" + ")
        result = f"\\left( {self.latex_operator.join(args_latex)} \\right)"
        if exp is not None:
            result = f"{result}^{{{printer.doprint(exp)}}}"
        return result
    
    def __format__(self, format_spec: str):
        # This forces f-strings to use your __str__ method, 
        # and then apply standard string formatting on top of that.
        return sp.sstr(self).__format__(format_spec)

class DeferredAdd(DeferredArithmetic):
    operator = "+"
    latex_operator = " + "

    def __new__(cls, *args):
        # Flatten nested DeferredAdd instances
        new_args = []
        for arg in args:
            if isinstance(arg, DeferredAdd):
                new_args.extend(arg.args)
            else:
                new_args.append(arg)
        
        # Extract rank and dim from the first element for validation
        # We assume the first element is a valid placeholder or array
        base = new_args[0]
        if check_if_scalar(base):
            ref_rank = 0
            ref_dim = []
        elif isinstance(base, NDimArray):
            dim_aux = list(base.shape)
            if sum(dim_aux) == 1:
                ref_rank = 0
                ref_dim = []
            else:
                ref_rank = base.rank()
                ref_dim = list(base.shape)
        else:
            ref_rank = getattr(base, 'rank', None)
            ref_dim = getattr(base, 'dim', None)
        
        if ref_rank is None:
            raise ValueError(f"Unknown rank for base element: {base}")
        
        # Validation
        for arg in new_args[1:]:
            # If we are dealing with a standard scalar symbol, assign rank 0
            if check_if_scalar(arg):
                arg_rank = 0
                arg_dim = []
            elif isinstance(arg, NDimArray):
                dim_aux = list(arg.shape)
                if sum(dim_aux) == 1:
                    arg_rank = 0
                    arg_dim = []
                else:
                    arg_rank = arg.rank()
                    arg_dim = list(arg.shape)
                arg_rank = arg.rank()
            else:
                arg_rank = getattr(arg, 'rank', None)
                arg_dim = getattr(arg, 'dim', None)
            
            if arg_rank is None:
                raise ValueError(f"Unknown rank for arg element: {arg}")
            if arg_rank != ref_rank:
                raise ValueError(f"Rank mismatch: {base} (rank {ref_rank}) vs {arg} (rank {arg_rank})")
            if arg_rank != 0 and arg_dim != ref_dim:
                raise ValueError(f"Dimension mismatch: {base} (dim {ref_dim}) vs {arg} (dim {arg_dim})")

        return sp.Expr.__new__(cls, *new_args)
    
    @property
    def rank(self):
        # All valid arguments share the same rank
        if check_if_scalar(self.args[0]):
            return 0
        elif isinstance(self.args[0], NDimArray):
            return self.args[0].rank()
        elif hasattr(self.args[0],'rank'):
            return self.args[0].rank
        else:
            raise TypeError("Rank of first argument not found")
        
    @property
    def dim(self):
        # All valid arguments share the same dim
        if check_if_scalar(self.args[0]):
            return []
        elif isinstance(self.args[0], NDimArray):
            dim_aux = list(self.args[0].shape)
            if sum(dim_aux) == 1:
                return []
            else:
                return dim_aux
        elif hasattr(self.args[0],'dim'):
            return self.args[0].dim
        else:
            raise TypeError("Dimensions of first argument not found")

    def _eval_subs(self, old, new):
        # Allow recursive substitution
        return DeferredAdd(*[arg.subs(old, new) if hasattr(arg, 'subs') else arg for arg in self.args])
    
    def evaluate(self):
        def fun_array(*args):
            return reduce(operator.add, args)
        return self._evaluate(fun_array, sp.Add)
        
        
class DeferredMul(DeferredArithmetic):
    operator = "*"
    latex_operator = " "

    def __new__(cls, *args):
        # Flatten nested DeferredMul instances
        new_args = []
        for arg in args:
            if isinstance(arg, DeferredMul):
                new_args.extend(arg.args)
            else:
                new_args.append(arg)
        
        # Rule: Only one "Tensor" allowed, others must be scalars
        ranks = cls._get_ranks_list(new_args)

        if None in ranks:
            raise ValueError(f"Unknown rank for one of the arguments: {new_args}")
        else:
            if sum(1 for r in ranks if r > 0) > 1:
                raise ValueError(f"Only one tensor allowed in multiplication: {new_args}")
            
        return sp.Expr.__new__(cls, *new_args)
    
    @property
    def rank(self):
        return max(self._get_ranks_list(self.args))
    
    @property
    def dim(self):
        for arg in self.args:
            if isinstance(arg, NDimArray):
                dim_aux = list(arg.shape)
                if sum(dim_aux) == 1:
                    return []
                else:
                    return dim_aux
            elif hasattr(arg,'dim') and len(arg.dim) > 0:
                return arg.dim
        return []
            
    
    def _eval_subs(self, old, new):
        # Allow recursive substitution
        return DeferredMul(*[arg.subs(old, new) if hasattr(arg, 'subs') else arg for arg in self.args])
    
    def evaluate(self):
        def fun_array(*args):
            return sp.prod(args)
        return self._evaluate(fun_array, sp.Mul)
        
class DeferredNegative(DeferredArithmetic):

    def __new__(cls, *args):
        if len(args) != 1:
            raise ValueError("Unary minus expects exactly one argument")
        return sp.Expr.__new__(cls, *args)
    
    @property
    def rank(self):
        # Rank of the only argument
        if check_if_scalar(self.args[0]):
            return 0
        elif isinstance(self.args[0], NDimArray):
            return self.args[0].rank()
        elif hasattr(self.args[0],'rank'):
            return self.args[0].rank
        else:
            raise TypeError("Rank of first argument not found")
        
    @property
    def dim(self):
        # Dim of the only argument
        if check_if_scalar(self.args[0]):
            return []
        elif isinstance(self.args[0], NDimArray):
            dim_aux = list(self.args[0].shape)
            if sum(dim_aux) == 1:
                return []
            else:
                return dim_aux
        elif hasattr(self.args[0],'dim'):
            return self.args[0].dim
        else:
            raise TypeError("Dimensions of first argument not found")
        
    def _eval_subs(self, old, new):
        # Allow recursive substitution
        arg = self.args[0]
        return DeferredNegative(arg.subs(old, new) if hasattr(arg, 'subs') else arg)
    
    def evaluate(self):
       def class_scalar(*args):
           return sp.Mul(sp.sympify(-1), *args)
       return self._evaluate(operator.neg, class_scalar)
        
    def _sympystr(self, printer, exp=None):
        # Print each argument. doprint() applies recursion.
        arg_str = printer.doprint(self.args[0])
        
        # Join them with the adequate operator
        # Parentehsis are added to prevent ambiguity
        result = f"(-{arg_str})"
        if exp is not None:
            result = f"{result}**{printer.doprint(exp)}"
        return result

    def _latex(self, printer, exp=None):
        # Print each argument via the LaTeX printer. doprint() applies recursion.
        arg_latex = printer.doprint(self.args[0])
        
        # Join using the LaTeX-specific operator (e.g., r" + ")
        result = f"\\left( - {arg_latex} \\right)"
        if exp is not None:
            result = f"{result}^{{{printer.doprint(exp)}}}"
        return result
        
class DeferredDivide(DeferredArithmetic):
    operator = "/"

    def __new__(cls, *args):
        if len(args) != 2:
            raise ValueError("Division expects exactly two arguments")
        
        # Rule: Denominator must be scalar
        ranks = cls._get_ranks_list(args)
        if None in ranks:
            raise ValueError(f"Unknown rank for one of the arguments: {args}")
        else:
            if ranks[1] > 0:
                raise ValueError(f"Denominator must always be scalar: {args}")
            
        return sp.Expr.__new__(cls, *args)
    
    @property
    def rank(self):
        return self._get_ranks_list(self.args)[0]
    
    @property
    def dim(self):
        if check_if_scalar(self.args[0]):
            return []
        elif isinstance(self.args[0], NDimArray):
            dim_aux = list(self.args[0].shape)
            if sum(dim_aux) == 1:
                return []
            else:
                return dim_aux
        elif hasattr(self.args[0],'dim'):
            return self.args[0].dim
        else:
            raise TypeError("Dimensions of first argument not found")
    
    def _eval_subs(self, old, new):
        # Allow recursive substitution
        return DeferredDivide(*[arg.subs(old, new) if hasattr(arg, 'subs') else arg for arg in self.args])
    
    def evaluate(self):
        def class_scalar(*args):
           if all([isinstance(a, sp.Number) for a in args]):
               return sp.Rational(*args)
           elif isinstance(args[1], sp.Number):
               return sp.Mul(args[0],sp.Rational(sp.sympify(1),args[1]))
           else:
               return operator.truediv(*args)
        return self._evaluate(operator.truediv, class_scalar)

    def _latex(self, printer, exp=None):
        # Print each argument via the LaTeX printer. doprint() applies recursion.
        args_latex = [printer.doprint(arg) for arg in self.args]
        
        # Join using the LaTeX-specific operator (e.g., r" + ")
        result = f"\\frac{"{"}{args_latex[0]}{"}"}{"{"}{args_latex[1]}{"}"}"
        if exp is not None:
            result = f"{result}^{{{printer.doprint(exp)}}}"
        return result
    
class DeferredPow(DeferredArithmetic):
    operator = "**"

    def __new__(cls, *args):
        if len(args) != 2:
            raise ValueError("Power expects exactly two arguments")
        
        # Rule: Both arguments must be scalar
        ranks = cls._get_ranks_list(args)
        if None in ranks:
            raise ValueError(f"Unknown rank for one of the arguments: {args}")
        else:
            if any([rank > 0 for rank in ranks]):
                raise ValueError(f"Base and exponent must always be scalar: {args}")
            
        return sp.Expr.__new__(cls, *args)
    
    @property
    def rank(self):
        return 0
    
    @property
    def dim(self):
        return []
    
    def _eval_subs(self, old, new):
        # Allow recursive substitution
        return DeferredPow(*[arg.subs(old, new) if hasattr(arg, 'subs') else arg for arg in self.args])
    
    def evaluate(self):
        return self._evaluate(operator.pow,sp.Pow)

    def _latex(self, printer, exp=None):
        # Print each argument via the LaTeX printer. doprint() applies recursion.
        args_latex = [printer.doprint(arg) for arg in self.args]
        
        # Join using the LaTeX-specific operator (e.g., r" + ")
        result = f"{args_latex[0]}^{"{"}{args_latex[1]}{"}"}"
        if exp is not None:
            result = f"{result}^{{{printer.doprint(exp)}}}"
        return result
        

class BaseTensorPlaceholder(sp.Expr):
    """
    This class represents a tensor placeholder, i.e. a symbolic object that has an associated tensor rank and dimension, among other properties, but does not have defined components.
    It enables the writing of variational formulations involving vectors, matrices and higher order tensors in a compact way close to pen and paper.
    The components of the tensor placeholder can be later generated by calling the generate_array() method, which creates a Sympy Array with the appropriate shape and fills it with new symbols.
    The original placeholder can then be substituted by its array in any expression using the substitute_array() method.
    """

    _op_priority = 100 # Give a vey high priority so deferred arithmetic operations are used whnever a BaseTensorPlaceholder is encountered (in left or right)

    array_name_complement = '_array'
    gauss_name_complement = '_gauss'

    # This indicates to sympy that it is commutative to + and * operators.
    # We will later restrict * to only allow multiplication by scalars and + to only objects of the same rank and dimensions.
    is_commutative = True
    
    # This indicates to sympy that this class should be treated as an atomic object, i.e. it should not be traversed
    # when doing substitutions or other operations.
    @property
    def is_Atom(self):
        return True
        # return False
    
    def __new__(cls, *args):
        return super().__new__(cls, *args)
    
    # We assign the properties of the placeholder as arguments of the sympy expression.
    @classmethod
    def from_info_dict(cls, info_dict):
        name = sp.core.symbol.Str(info_dict['symbol']) # Prevent it from being converted to a symbol
        rank = sp.sympify(info_dict['tensor_rank']) # Should be a sp.Integer
        dim = sp.sympify(list(info_dict['dim']))
        dim_str = ",".join([str(i) for i in list(info_dict['dim'])]) # We turn it into a string to make it hashable as an arg
        dim_str = sp.core.symbol.Str(dim_str)
        if len(dim)!= rank:
            raise ValueError(f'Dimensions {dim} for {name} are not compatible with rank {rank}.')
        dependencies = info_dict['dependencies']
        if isinstance(dependencies, str):
            dependencies = sp.core.symbol.Str(dependencies) # Prevent it from beings converted to a symbol
        latex_str = sp.core.symbol.Str(info_dict.get('latex', name)) # Prevent it from beings converted to a symbol
        flags = []
        if info_dict.get('symmetric', False):
            flags.append('symmetric')
        if info_dict.get('third_symmetry', False):
            flags.append('third_symmetry')
        if info_dict.get('use_voigt_notation', False):
            flags.append('use_voigt_notation')
        if info_dict.get('positive', False):
            flags.append('positive')
        flags = sp.core.symbol.Str(",".join(flags))
        return cls(name, rank, dim_str, dependencies, latex_str, flags)
    
    # Then we create accessors for those properties.
    @property
    def name(self):
        return str(self.args[0])
    
    @property
    def rank(self):
        return self.args[1]
    
    @property
    def dim(self):
        # We recover the list from the string arg
        dim_str = str(self.args[2])
        if dim_str:
            return [int(i) for i in dim_str.split(",")]
        else:
            return []
    
    @property
    def dependencies(self):
        return self.args[3]
    
    @property
    def latex_str(self):
        return str(self.args[4])
    
    @property
    def flags(self):
        flags_str = str(self.args[5])
        if flags_str:
            return [flag for flag in flags_str.split(",")]
        else:
            return []
    
    @property
    def array(self):
        array_name = self.name+self.array_name_complement
        return self._generate_functions_array(array_name, self.dependencies)
        
    def substitute_components_simulatneous(self, expr, original, new):
        original_iterator = get_flat_list_of_components(original)
        new_iterator = get_flat_list_of_components(new)
        assert len(original_iterator)==len(new_iterator)
        subs_dict = dict(zip(original_iterator, new_iterator))
        return expr.subs(subs_dict, simultaneous=True)

    # Algebraic operators
    def __add__(self, other):
        # return add_op(self, other)
        return DeferredAdd(self, other)
    
    def __radd__(self, other):
        # return add_op(other, self)
        return DeferredAdd(other, self)
    
    def __neg__(self):
        return DeferredNegative(self)
    
    def __mul__(self, other):
        # return prod_op(self, other)
        return DeferredMul(self, other)
    
    def __rmul__(self, other):
        # return prod_op(other, self)
        return DeferredMul(other, self)
    
    def __truediv__(self, other):
        return DeferredDivide(self, other)
    
    def __rtruediv__(self, other):
        return DeferredDivide(other, self)
    
    def __pow__(self, other):
        return DeferredPow(self, other)
    
    def __rpow__(self, other):
        return DeferredPow(other, self)
    
    def _latex(self, printer, exp=None, *args):
        if isinstance(printer, CustomLatexPrinter):
            # symbol, rank, dim = [printer._print(i) for i in self.args]
            if exp is None:
                return r"%s" % (self.latex_str)
            else:
                return r"%s^{%s}" % (self.latex_str, exp)
        else:
            return printer._print_Function(self, exp=exp)

    def _sympystr(self, printer):
        # How it looks in a standard print()
        return self.name

    def _pretty(self, printer):
        # Unicode implementation
        return sp.pretty_form.PrettyString(self.name)
    
    def __repr__(self):
        class_name = self.__class__.__name__
        return f"{class_name}({self.name}, dim={self.dim}, rank={self.rank})"

    def __format__(self, format_spec: str):
        # This forces f-strings to use your __str__ method, 
        # and then apply standard string formatting on top of that.
        return sp.sstr(self).__format__(format_spec)
    
    def _fill_array_rank2(self, base_name, dependencies_list):
        flags = [str(flag) for flag in self.flags] # Convert to strings for comparison
        number_of_elements = 1
        for dim_comp in self.dim:
            number_of_elements *= dim_comp
        out_array = sp.MutableDenseNDimArray(sp.zeros(number_of_elements),shape=self.dim)
        for i in range(self.dim[0]):
            for j in range(self.dim[1]):
                if "symmetric" in flags:
                    indexes_string = '_'+str(min(i,j))+'_'+str(max(i,j)) # Apply symmetry
                else:
                    indexes_string = '_'+str(i)+'_'+str(j)
                if dependencies_list is None:
                    out_array[i,j] = sp.Symbol(base_name+indexes_string)
                else:
                    out_array[i,j] = sp.Function(base_name+indexes_string)(*dependencies_list)
        return out_array
    
    def _fill_array_rank4(self, base_name, dependencies_list):
        flags = [str(flag) for flag in self.flags] # Convert to strings for comparison
        number_of_elements = 1
        for dim_comp in self.dim:
            number_of_elements *= dim_comp
        out_array = sp.MutableDenseNDimArray(sp.zeros(number_of_elements),shape=self.dim)
        for i in range(self.dim[0]):
            for j in range(self.dim[1]):
                for k in range(self.dim[2]):
                    for l in range(self.dim[3]):
                        unique_index = [min(i,j), max(i,j), min(k,l), max(k,l)] # Apply first two symmetries
                        if "third_symmetry" in flags:
                            if unique_index[0] > unique_index[2]:
                                unique_index = [unique_index[2],unique_index[3],unique_index[0],unique_index[1]]
                            elif unique_index[0] == unique_index[2]:
                                if unique_index[1] > unique_index[3]:
                                    unique_index = [unique_index[2],unique_index[3],unique_index[0],unique_index[1]]
                        indexes_string=''
                        for index in unique_index:
                            indexes_string += '_'+str(index)
                        if dependencies_list is None:
                            out_array[i,j,k,l] = sp.Symbol(base_name+indexes_string)
                        else:
                            out_array[i,j,k,l] = sp.Function(base_name+indexes_string)(*dependencies_list)
        return out_array
    
    def _generate_functions_array(self, base_name, dependencies_list):
        if self.rank == 0:
            if dependencies_list is None:
                return sp.Symbol(base_name)
            else:
                return sp.Function(base_name)(*dependencies_list)
        elif self.rank == 1:
            if dependencies_list is None:
                return sp.Array([sp.Symbol(base_name+'_'+str(i)) for i in range(self.dim[0])])
            else:
                return sp.Array([sp.Function(base_name+'_'+str(i))(*dependencies_list) for i in range(self.dim[0])])
        elif self.rank == 2:
            return self._fill_array_rank2(base_name, dependencies_list)
        elif self.rank == 4:
            return self._fill_array_rank4(base_name, dependencies_list)
        else:
            raise NotImplementedError(f"Rank {self.rank} not implemented yet for _generate_functions_array() in BaseTensorPlaceholder")


class VarsCombination(sp.Expr):
    """
    This is a container for combinations of variables to be used as dependecies or as arguments for derivatives.
    This base class implements the array property by returning a list of all the flatttened components of the arguments in their array form.

    Example: VarsCombination(A,x) where A is a SymbolicPlaceholder of rank 2 and x is a Sympy Symbol.
        Then VarsCombination(A,x).array will return [A_0_0, A_0_1, A_1_0, A_1_1, x]
    """

    is_commutative = False
    
    @property
    def is_Atom(self):
        return True
    
    def __new__(cls, *components):
        for comp in components:
            if not isinstance(comp, sp.Basic):
                raise TypeError(f"Cannot include {comp} in a VarsCombination, as  it is not a Sympy object")
        return sp.Expr.__new__(cls, *components)
    
    @classmethod
    def from_namespace(cls, args, namespace):
        # Initialize a VarsCombination based on a list of names and a namespace
        components = [namespace[name] for name in args]
        return cls(*components)
    
    def _eval_subs(self, old, new):
        # No recursive substitution
        return self
    
    @property
    def rank(self):
        return 1
    
    @property
    def dim(self):
        total_dims = 0
        for arg in self.args:
            if isinstance(arg, BaseTensorPlaceholder):
                res = 1
                for val in arg.dim:
                    res = res * val
                total_dims += res
            elif isinstance(arg, sp.NDimArray):
                res = 1
                for val in arg.shape:
                    res = res * val
                total_dims += res
            elif isinstance(arg, sp.Symbol) or isinstance(arg, sp.Function):
                total_dims += 1
            else:
                raise Exception(f"Object {arg} within VarsCombination is not recognized")
        return [total_dims]
    
    @property
    def array(self):
        flattened_components = []
        for arg in self.args:
            # If any of the arguments is a BaseTensorPlaceholder, get its array attribute and flatten it
            if isinstance(arg, BaseTensorPlaceholder):
                flattened_components.extend(get_flat_list_of_components(arg.array))
            else:
                flattened_components.extend(get_flat_list_of_components(arg))
        return flattened_components
    
    @property
    def gauss(self):
        flattened_components = []
        for arg in self.args:
            # If any of the arguments is a BaseTensorPlaceholder, get its array attribute and flatten it
            if isinstance(arg, BaseTensorPlaceholder):
                flattened_components.extend(get_flat_list_of_components(arg.gauss))
            else:
                flattened_components.extend(get_flat_list_of_components(arg))
        return flattened_components
    
    def __len__(self):
        return len(self.array)
    
    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count < len(self):
            out = self.array[self.count]
            self.count += 1
            return out
        raise StopIteration
    
    def get_tuple_string(self, printer=None):
        out_str = '('
        for var in self.args:
            if isinstance(var,BaseTensorPlaceholder):
                out_str += var.latex_str+','
            elif printer is not None:
                out_str += printer._print(var)+','
            else:
                out_str += str(var)+','
        out_str = out_str[:-1]+')'
        return out_str

    def get_names_list(self):
        return [arg.name for arg in self.args]
    
    def __repr__(self):
        return "VarsCombination("+",".join([repr(arg) for arg in self.args])+")"
    

class CoordsIndicator(VarsCombination):
    
    def __new__(cls, *components):
        for comp in components:
            if not isinstance(comp, sp.Symbol):
                raise TypeError(f"Cannot include {comp} in a CoordsIndicator, as  it is not a Sympy Symbol")
        return sp.Expr.__new__(cls, *components)
    
    @property
    def rank(self):
        return 1
    
    @property
    def dim(self):
        # All components must be scalars, so the total dimension is just the number of components
        return [self.__len__()]
    
    @property
    def array(self):
        return list(self.args)
    
    @property
    def gauss(self):
        raise NotImplementedError("CoordsIndicator does not implement gauss attribute")
    
    def __len__(self):
        return len(self.args)
    
    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count < len(self):
            out = self.args[self.count]
            self.count += 1
            return out
        raise StopIteration
    
    def get_tuple_string(self, printer=None):
        out_str = '('
        for var in self.args:
            if printer is not None:
                out_str += printer._print(var)+','
            else:
                out_str += str(var)+','
        out_str = out_str[:-1]+')'
        return out_str
    
    def __repr__(self):
        return "CoordsIndicator("+",".join([repr(arg) for arg in self.args])+")"

class CoefficientsIndicator(VarsCombination):
    """
    Used as a possible dependency of tensors used in the substitution phases. Refers to the nodal coefficients of the given variables.
    """

    def __new__(cls, *components):
        for comp in components:
            if isinstance(comp, BaseTensorPlaceholder) and not comp.__class__.__name__ in ['NodalTensorPlaceholder','UnknownTensorPlaceholder']:
                raise TypeError(f"Cannot include {comp} in a VarsCombination, as  it is not a NodalTensorPlaceholder or UnknownTensorPlaceholder")
        return sp.Expr.__new__(cls, *components)
    
    @property
    def array(self):
        # raise NotImplementedError("CoefficientsIndicator does not implement array attribute. It is meant to be used as a possible dependency of tensors used in the substitution phases.")
        return self
    
    @property
    def gauss(self):
        # Returns the list of nodal values for all components.
        # The order is [var_1_comp_1_node_1, var_1_comp_2_node_1, var_2_node_1, var_1_comp_1_node_2, var_1_comp_2_node_2, var_2_node_2, ...]
        combined_dofs_mat = self.args[0].nodes
        for var in self.args[1:]:
            combined_dofs_mat = combined_dofs_mat.row_join(var.nodes)
        return get_flat_list_of_components(combined_dofs_mat)
    
    def evaluate(self):
        # Unlike VarsCombination and CoordsIndicator, this one can be used as a standalone input to a DerivIndicator's denominator.
        # Therefore it implements an evaluate function that just returns the gauss property (variable nodal values) in array form.
        return sp.Array(self.gauss)
    
    def __len__(self):
        return 1
    
    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count < len(self):
            self.count += 1
            return self.array
        raise StopIteration
    
    def __repr__(self):
        return "CoefficientsIndicator("+",".join([repr(arg) for arg in self.args])+")"

class DeferredTensorOp(sp.Expr):
    _op_priority = 100 # Give a vey high priority so deferred arithmetic operations are used whnever a DeferredTensorOp is encountered (in left or right)

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
    
    def __neg__(self):
        return DeferredNegative(self)
    
    def __mul__(self, other):
        # return prod_op(self, other)
        return DeferredMul(self, other)
    
    def __rmul__(self, other):
        # return prod_op(other, self)
        return DeferredMul(other, self)
    
    def __truediv__(self, other):
        return DeferredDivide(self, other)
    
    def __rtruediv__(self, other):
        return DeferredDivide(other, self)
    
    def __pow__(self, other):
        return DeferredPow(self, other)
    
    def __rpow__(self, other):
        return DeferredPow(other, self)
    
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