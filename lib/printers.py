from sympy.printing.latex import LatexPrinter

class CustomLatexPrinter(LatexPrinter):
    pass
    
def print_my_latex(expr):
    """ Most of the printers define their own wrappers for print().
    These wrappers usually take printer settings. Our printer does not have
    any settings.
    """
    return CustomLatexPrinter().doprint(expr)