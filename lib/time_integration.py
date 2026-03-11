import sympy as sp

from lib.placeholder_operators import add_op, prod_op
from lib.basic_classes import BaseTensorPlaceholder
from lib.printers import CustomLatexPrinter

class TimeDerivativeBDF2(sp.Function):
    @classmethod
    def eval(cls, tensor_0, tensor_1, tensor_2, coeffs):

        if (not isinstance(tensor_0, BaseTensorPlaceholder) and not isinstance(tensor_1, BaseTensorPlaceholder) and
            not isinstance(tensor_2, BaseTensorPlaceholder)):
            return add_op(prod_op(coeffs[0],tensor_0), prod_op(coeffs[1], tensor_1), prod_op(coeffs[2], tensor_2))
            # raise TypeError('original_tensor should be either an sp.Array or a TensorPlaceholder')

    def _latex(self, printer, exp=None, *args):
        if isinstance(printer, CustomLatexPrinter):
            tensor_0, tensor_1, tensor_2, coeffs = [printer._print(i) for i in self.args]
            if exp is None:
                return r"\frac{\partial %s}{\partial t}" % (tensor_0)
            else:
                return r"\left(\frac{\partial %s}{\partial t}\right)^{%s}" % (tensor_0, exp)
        else:
            return printer._print_Function(self, exp=exp)