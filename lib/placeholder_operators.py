import sympy as sp

from lib.basic_classes import BaseTensorPlaceholder
from lib.printers import CustomLatexPrinter
from sympy.printing.precedence import PRECEDENCE, PRECEDENCE_TRADITIONAL
from lib.tensor_operators import *


## Define array-compatible versions of most basic operators

class add_op(sp.Function):
    # precedence = PRECEDENCE["Add"]
    # is_Add = True

    @classmethod
    def eval(cls, *args):
        do_eval = True
        for arg in args:
            if arg.has(BaseTensorPlaceholder):
                do_eval = False
        if do_eval:
            rv = args[0]
            for arg in args[1:]:
                rv += arg
            return rv
        
    def doit(self, deep=False, **hints):
        exp_args = []
        if deep:
            for arg in self.args:
                exp_args.append(arg.doit(deep=deep, **hints))
        else: 
            exp_args = self.args
        rv = exp_args[0]
        for exp_arg in exp_args[1:]:
            rv += exp_arg
        return rv

class prod_op(sp.Function):
    # precedence = PRECEDENCE['Mul']
    # is_Mul = True

    @classmethod
    def eval(cls, a, b):
        if not a.has(BaseTensorPlaceholder) and not b.has(BaseTensorPlaceholder):
            return a*b
        
    def doit(self, deep=False, **hints):
        a, b = self.args
        if deep:
           a, b = a.doit(deep=deep, **hints), b.doit(deep=deep, **hints)
        return a*b

class neg_op(sp.Function):

    @classmethod
    def eval(cls, a):
        return prod_op(-1,a)

class sub_op(sp.Function):

    @classmethod
    def eval(cls, a, b):
        return add_op(a,neg_op(b))

## Define tensorial operators
class norm_op(sp.Function):

    @classmethod
    def eval(cls, a):
        if not a.has(BaseTensorPlaceholder):
            return sp.sqrt(tensor_dot(a,a))
        
    def _latex(self, printer, exp=None, *args):
        if isinstance(printer, CustomLatexPrinter):
            a = printer._print(self.args[0])
            if exp is None:
                return r"\lVert %s \rVert" % (a)
            else:
                return r"\lVert %s \rVert^{%s}" % (a, exp)
        else:
            return printer._print_Function(self, exp=exp)

class dot_op(sp.Function):
    precedence = PRECEDENCE_TRADITIONAL['Dot']

    @classmethod
    def eval(cls, a, b):
        if not a.has(BaseTensorPlaceholder) and not b.has(BaseTensorPlaceholder):
            return tensor_dot(a,b)
        
    def _latex(self, printer, exp=None, *args):
        if isinstance(printer, CustomLatexPrinter):
            a, b = self.args
            rv = r"%s \cdot %s" % (printer.parenthesize(a, PRECEDENCE['Mul']),
                                 printer.parenthesize(b, PRECEDENCE['Mul']))
            if exp is None:
                return rv
            else:
                return r"\left(%s\right)^{%s}" % (rv, exp)
        else:
            return printer._print_Function(self, exp=exp)
        
class contract_op(sp.Function):
    precedence = PRECEDENCE_TRADITIONAL['Dot']

    @classmethod
    def eval(cls, a, b):
        if not a.has(BaseTensorPlaceholder) and not b.has(BaseTensorPlaceholder):
            return tensor_singlecontract(a,b)
    
    def _latex(self, printer, exp=None, *args):
        if isinstance(printer, CustomLatexPrinter):
            a, b = self.args
            rv = r"%s \cdot %s" % (printer.parenthesize(a, PRECEDENCE['Mul']),
                                 printer.parenthesize(b, PRECEDENCE['Mul']))
            if exp is None:
                return rv
            else:
                return r"\left(%s\right)^{%s}" % (rv, exp)
        else:
            return printer._print_Function(self, exp=exp)

class doublecontract_op(sp.Function):
    precedence = PRECEDENCE_TRADITIONAL['Dot']

    @classmethod
    def eval(cls, a, b):
        if not a.has(BaseTensorPlaceholder) and not b.has(BaseTensorPlaceholder):
            return tensor_doublecontract(a,b)
        
    def _latex(self, printer, exp=None, *args):
        if isinstance(printer, CustomLatexPrinter):
            a, b = self.args
            rv = r"%s : %s" % (printer.parenthesize(a, PRECEDENCE['Mul']),
                                 printer.parenthesize(b, PRECEDENCE['Mul']))
            if exp is None:
                return rv
            else:
                return r"\left(%s\right)^{%s}" % (rv, exp)
        else:
            return printer._print_Function(self, exp=exp)
        
class grad_op(sp.Function): # Returns dfj/dxi
    precedence = PRECEDENCE_TRADITIONAL['Gradient']

    @classmethod
    def eval(cls, a, base_scalars, transposed_gradients):
        if not a.has(BaseTensorPlaceholder) and not base_scalars.has(BaseTensorPlaceholder):
            if transposed_gradients:
                return tensor_transpose_grad(a,base_scalars)
            else:
                return tensor_grad(a,base_scalars)
        
    def _latex(self, printer, exp=None, *args):
        if isinstance(printer, CustomLatexPrinter):
            a, base_scalars, transposed_gradients = self.args
            rv = r"\nabla %s" % printer.parenthesize(a, PRECEDENCE['Mul'])
            if exp is None:
                return rv
            else:
                return r"\left(%s\right)^{%s}" % (rv, exp)
        else:
            return printer._print_Function(self, exp=exp)

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