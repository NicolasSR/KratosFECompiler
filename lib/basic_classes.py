import sympy as sp

from lib.components_utils import get_flat_list_of_components
from lib.printers import CustomLatexPrinter

class BaseTensorPlaceholder(sp.Function):
    @classmethod
    def eval(cls, name):
        pass

    def apply_tensor_config(self, info_dict):
        assert self.args[0].name == info_dict['symbol']
        self.symbol = self.args[0]
        self.name = self.symbol.name
        self.dim = info_dict['dim']
        self.latex_str = info_dict.get('latex', self.name)

    def substitute_components_simulatneous(self, expr, original, new):
        original_iterator = get_flat_list_of_components(original)
        new_iterator = get_flat_list_of_components(new)
        assert len(original_iterator)==len(new_iterator)
        subs_dict = dict(zip(original_iterator, new_iterator))
        return expr.subs(subs_dict, simultaneous=True)
    
    def _latex(self, printer, exp=None, *args):
        if isinstance(printer, CustomLatexPrinter):
            # symbol, rank, dim = [printer._print(i) for i in self.args]
            if exp is None:
                return r"%s" % (self.latex_str)
            else:
                return r"%s^{%s}" % (self.latex_str, exp)
        else:
            return printer._print_Function(self, exp=exp)

class VarsCombination():
    def __init__(self, *args):
        self.vars_symbols = list(args)
        self.vars_objects = []

    def update_symbol_objects(self, SYMB):
        self.vars_objects = [SYMB[name] for name in self.vars_symbols]
        
    def get_tuple_string(self, printer=None):
        out_str = '('
        for var in self.vars_objects:
            if isinstance(var,BaseTensorPlaceholder):
                out_str += var.latex_str+','
            elif printer is not None:
                out_str += printer._print(var)+','
            else:
                out_str += str(var)+','
        out_str = out_str[:-1]+')'

        return out_str
    
    def get_names_list(self):
        return self.vars_symbols
    
    def get_dependency_list_array(self, SYMB, name_complement):
        return self.get_flat_objects_list(SYMB, name_complement)
    
    def get_dependency_list_array_or_matrix(self, SYMB, name_complement):
        return self.get_flat_objects_list(SYMB, name_complement)

    def get_flat_objects_list(self, SYMB, name_complement):
        out = []
        for symbol in self.vars_symbols:
            array = SYMB[symbol+name_complement]
            out.extend(get_flat_list_of_components(array))
        return tuple(out)
    
    def get_array(self):
        return sp.Array(self.vars_objects)
    
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