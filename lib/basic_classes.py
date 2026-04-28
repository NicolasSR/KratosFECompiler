from functools import reduce
import operator
import itertools

import sympy as sp
from sympy import NDimArray

from lib.components_utils import get_flat_list_of_components
from lib.printers import CustomLatexPrinter
from lib.coordinates_system import ACTIVE_COORD_SYSTEM

def check_if_scalar(arg):
    return  str(sp.sympify(arg).kind) == "NumberKind" or isinstance(arg, sp.Function)

class DeferredArithmetic(sp.Expr):
    """
    Common base class for DeferredAdd and DeferredMul, to handle arithmetics and printing.
    """

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
            ref_dim = None
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
                arg_dim = None
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
            return list(self.args[0].shape)
        elif hasattr(self.args[0],'dim'):
            return self.args[0].dim
        else:
            raise TypeError("Dimensions of first argument not found")

    def _eval_subs(self, old, new):
        # Allow recursive substitution
        return DeferredAdd(*[arg.subs(old, new) if hasattr(arg, 'subs') else arg for arg in self.args])
    
    def evaluate(self):
        # Recursively evaluate children first
        evaluated_args = [arg.evaluate() if hasattr(arg, 'evaluate') else arg 
                          for arg in self.args]
        
        if any(isinstance(a, BaseTensorPlaceholder) for a in evaluated_args):
            return self
        elif any(isinstance(a, NDimArray) for a in evaluated_args):
            # return sum(evaluated_args) # Let SymPy handle array addition
            return reduce(operator.add, evaluated_args)
        else:
            return sp.Add(*evaluated_args) # Fallback to standard SymPy for scalars
        
        
class DeferredMul(DeferredArithmetic):
    operator = "*"
    latex_operator = " "

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
            elif hasattr(arg,'dim') and arg.dim is not None:
                return arg.dim
        return []
            
    
    def _eval_subs(self, old, new):
        # Allow recursive substitution
        return DeferredMul(*[arg.subs(old, new) if hasattr(arg, 'subs') else arg for arg in self.args])
    
    def evaluate(self):
        # Recursively evaluate children first
        evaluated_args = [arg.evaluate() if hasattr(arg, 'evaluate') else arg 
                          for arg in self.args]
        
        if any(isinstance(a, BaseTensorPlaceholder) for a in evaluated_args):
            return self
        elif any(isinstance(a, NDimArray) for a in evaluated_args):
            return sp.prod(evaluated_args) # Let SymPy handle array multiplication
        else:
            return sp.Mul(*evaluated_args) # Fallback to standard SymPy for scalars
        

class BaseTensorPlaceholder(sp.Expr):
    """
    This class represents a tensor placeholder, i.e. a symbolic object that has an associated tensor rank and dimension, among other properties, but does not have defined components.
    It enables the writing of variational formulations involving vectors, matrices and higher order tensors in a compact way close to pen and paper.
    The components of the tensor placeholder can be later generated by calling the generate_array() method, which creates a Sympy Array with the appropriate shape and fills it with new symbols.
    The original placeholder can then be substituted by its array in any expression using the substitute_array() method.
    """

    # This indicates to sympy that it is commutative to + and * operators.
    # We will later restrict * to only allow multiplication by scalars and + to only objects of the same rank and dimensions.
    is_commutative = True
    
    # This indicates to sympy that this class should be treated as an atomic object, i.e. it should not be traversed
    # when doing substitutions or other operations.
    @property
    def is_Atom(self):
        return True
    
    # We assign the properties of the placeholder as arguments of the sympy expression.
    def __new__(cls, info_dict):
        name = info_dict['symbol']
        rank = info_dict['tensor_rank']
        dim = list(info_dict['dim'])
        if len(dim)!= rank:
            raise ValueError(f'Dimensions {dim} for {name} are not compatible with rank {rank}.')
        dependencies = info_dict['dependencies']
        latex_str = info_dict.get('latex', name)
        return sp.Expr.__new__(cls, name, rank, dim, dependencies, latex_str)
    
    # Then we create accessors for those properties.
    @property
    def name(self):
        return self.args[0]
    
    @property
    def rank(self):
        return self.args[1]
    
    @property
    def dim(self):
        return self.args[2]
    
    @property
    def dependencies(self):
        return self.args[3]
    
    @property
    def latex_str(self):
        return self.args[4]
        
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
    
    def __mul__(self, other):
        # return prod_op(self, other)
        return DeferredMul(self, other)
    
    def __rmul__(self, other):
        # return prod_op(other, self)
        return DeferredMul(other, self)
    
    # Printing
    
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
        return f"BaseTensorPlaceholder({self.name}, dim={self.dim}, rank={self.rank})"

    def __format__(self, format_spec: str):
        # This forces f-strings to use your __str__ method, 
        # and then apply standard string formatting on top of that.
        return sp.sstr(self).__format__(format_spec)
    
    ## TO BE DELETED
    def generate_array(self):
        self.array_name = self.name+'_array'
        dependencies_list = ACTIVE_COORD_SYSTEM.get()["coord_symbols"]
        if self.rank == 0:
            self.array = sp.Function(self.array_name)(*dependencies_list)
        if self.rank == 1:
            self.array = sp.Array([sp.Function(self.array_name+'_'+str(i))(*dependencies_list) for i in range(self.dim[0])])

class VarsCombination(sp.Expr):

    is_commutative = False
    
    @property
    def is_Atom(self):
        return False
    
    # We assign the properties of the placeholder as arguments of the sympy expression.
    def __new__(cls, *components):
        for comp in components:
            if not isinstance(comp, sp.Basic):
                raise TypeError(f"Cannot include {comp} in a VarsCombination, as  it is not a Sympy object")
        return sp.Expr.__new__(cls, *components)

    def evaluate(self):
        # Return array of flattened components. We assime everything inside will be either NDimArrays or Functions or Symbols
        flattened_components = []
        for arg in self:
            if isinstance(arg, sp.NDimArray):
                dim = list(arg.shape)
                index_ranges = [range(d) for d in dim]
                index_combinations = itertools.product(*index_ranges)
                flattened_components.extend([arg[indices] for indices in index_combinations])
            else:
                flattened_components.append(arg)
        return sp.Array(flattened_components)
    
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
    
    

# class VarsCombination():
#     def __init__(self, *args):
#         self.vars_symbols = list(args)
#         self.vars_objects = []

#     def update_symbol_objects(self, SYMB):
#         self.vars_objects = [SYMB[name] for name in self.vars_symbols]
        
#     def get_tuple_string(self, printer=None):
#         out_str = '('
#         for var in self.vars_objects:
#             if isinstance(var,BaseTensorPlaceholder):
#                 out_str += var.latex_str+','
#             elif printer is not None:
#                 out_str += printer._print(var)+','
#             else:
#                 out_str += str(var)+','
#         out_str = out_str[:-1]+')'

#         return out_str
    
#     def get_names_list(self):
#         return self.vars_symbols
    
#     def get_dependency_list_array(self, SYMB, name_complement):
#         return self.get_flat_objects_list(SYMB, name_complement)
    
#     def get_dependency_list_array_or_matrix(self, SYMB, name_complement):
#         return self.get_flat_objects_list(SYMB, name_complement)

#     def get_flat_objects_list(self, SYMB, name_complement):
#         out = []
#         for symbol in self.vars_symbols:
#             array = SYMB[symbol+name_complement]
#             out.extend(get_flat_list_of_components(array))
#         return tuple(out)
    
#     def get_array(self):
#         return sp.Array(self.vars_objects)
    
class DofsIndicator(VarsCombination):
    
    def get_dependency_list_array_or_matrix(self, SYMB, name_complement=''):
        out = []
        nnodes = SYMB[self.vars_symbols[0]+'_nodes'].shape[0]
        for node in range(nnodes):
            for symbol in self.vars_symbols:
                assert SYMB[symbol+'_nodes'].shape[0]==nnodes
                array = SYMB[symbol+'_nodes'][node,:]
                out.extend(get_flat_list_of_components(array))
        return tuple(out)
    
class DerivIndicator():
    def __init__(self, function, vars):
        self.function_symbol = function
        self.vars_symbol = vars

    def get_args_names_list(self):
        out = [self.function_symbol]
        if isinstance(self.vars_symbol, VarsCombination):
            out.extend(self.vars_symbol.get_names_list())
        elif isinstance(self.vars_symbol,str):
            out.append(self.vars_symbol)
        else:
            raise "Only acceptable formats for derivative denominator are string, VarsCombination() or DofsIndicator()"
        return out

    def get_deriv_name(self):
        return '_'.join(self.get_args_names_list())+'_deriv'
    
    def get_derivative_iterator(self, SYMB, name_complement, transposed):
        numerator_iterator = get_flat_list_of_components(SYMB[self.function_symbol+name_complement])
        if isinstance(self.vars_symbol, DofsIndicator):
            denominator_iterator = self.vars_symbol.get_dependency_list_array_or_matrix(SYMB)
        elif isinstance(self.vars_symbol, VarsCombination):
            denominator_iterator = self.vars_symbol.get_objects_list(SYMB, name_complement)
        elif isinstance(self.vars_symbol,str):
            denominator_iterator = get_flat_list_of_components(SYMB[self.vars_symbol+name_complement])
        derivatives_list = []
        if transposed:
            for numerator in numerator_iterator:
                for denominator in denominator_iterator:
                    derivatives_list.append(sp.Derivative(numerator,denominator))
        else:
            for denominator in denominator_iterator:
                for numerator in numerator_iterator:
                    derivatives_list.append(sp.Derivative(numerator,denominator))
        return derivatives_list
    
    def get_derivative_iterator_with_substitutions(self,SYMB, name_complement, transposed, subs_function):
        numerator_iterator_aux = get_flat_list_of_components(SYMB[self.function_symbol+name_complement])
        numerator_iterator = [subs_function(numerator_elem_aux) for numerator_elem_aux in numerator_iterator_aux]
        if isinstance(self.vars_symbol, DofsIndicator):
            denominator_iterator = self.vars_symbol.get_dependency_list_array_or_matrix(SYMB)
        elif isinstance(self.vars_symbol, VarsCombination):
            denominator_iterator_aux = self.vars_symbol.get_objects_list(SYMB, name_complement)
            denominator_iterator = [subs_function(denominator_elem_aux) for denominator_elem_aux in denominator_iterator_aux]
        elif isinstance(self.vars_symbol,str):
            denominator_iterator_aux = get_flat_list_of_components(SYMB[self.vars_symbol+name_complement])
            denominator_iterator = [subs_function(denominator_elem_aux) for denominator_elem_aux in denominator_iterator_aux]
        derivatives_list = []
        if transposed:
            for numerator in numerator_iterator:
                for denominator in denominator_iterator:
                    derivatives_list.append(sp.Derivative(numerator,denominator))
        else:
            for denominator in denominator_iterator:
                for numerator in numerator_iterator:
                    derivatives_list.append(sp.Derivative(numerator,denominator))
        return derivatives_list

# ## Define tensorplaceholder-compatible versions of most basic operators

# class add_op(sp.Function):
#     # precedence = PRECEDENCE["Add"]
#     # is_Add = True

#     @classmethod
#     def eval(cls, *args):
#         do_eval = True
#         for arg in args:
#             if arg.has(BaseTensorPlaceholder):
#                 do_eval = False
#         if do_eval:
#             rv = args[0]
#             print("evaling add_op with args: ", [type(arg) for arg in args])
#             for arg in args[1:]:
#                 rv += arg
#             print('done')
#             return rv
        
#     def doit(self, deep=False, **hints):
#         exp_args = []
#         if deep:
#             for arg in self.args:
#                 exp_args.append(arg.doit(deep=deep, **hints))
#         else: 
#             exp_args = self.args
#         rv = exp_args[0]
#         for exp_arg in exp_args[1:]:
#             rv += exp_arg
#         return rv

# class prod_op(sp.Function):
#     # precedence = PRECEDENCE['Mul']
#     # is_Mul = True

#     @classmethod
#     def eval(cls, a, b):
#         if not a.has(BaseTensorPlaceholder) and not b.has(BaseTensorPlaceholder):
#             return a*b
        
#     def doit(self, deep=False, **hints):
#         a, b = self.args
#         if deep:
#            a, b = a.doit(deep=deep, **hints), b.doit(deep=deep, **hints)
#         return a*b

# class neg_op(sp.Function):

#     @classmethod
#     def eval(cls, a):
#         return prod_op(-1,a)

# class sub_op(sp.Function):

#     @classmethod
#     def eval(cls, a, b):
#         return add_op(a,neg_op(b))