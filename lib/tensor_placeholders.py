import numpy as np
import sympy as sp

from lib.basic_classes import BaseTensorPlaceholder, VarsCombination, DerivIndicator
from lib.array_definitions import DefineVector, DefineMatrix, DefineSymmetricMatrix
from lib.kratos_utilities import DfjDxi
from lib.printers import CustomLatexPrinter
from lib.utilities import substitute_symbols, substitute_functions

VOIGT_INDEX_DICT = {
    2: [[0,0],[1,1],[0,1]],
    # 3: [[0,0],[1,1],[2,2],[1,2],[0,2],[0,1]] # Standard Voigt notation
    3: [[0,0],[1,1],[2,2],[0,1],[1,2],[0,2]]  # Kratos Voigt notation
    }

def disambiguate_var_group(var_group, SYMB):
    if isinstance(var_group, str):
        out = VarsCombination(var_group)
        out.update_symbol_objects(SYMB)
        return out
    elif isinstance(var_group, DerivIndicator):
        return var_group
    elif isinstance(var_group, VarsCombination):
        return var_group
    else:
        raise ValueError("Invalid input type for disambiguate_var_group()")

class FunctionTensorPlaceholder(BaseTensorPlaceholder):
    array_name_complement = '_array'
    gauss_name_complement = '_gauss'

    def apply_tensor_config(self, info_dict):
        super().apply_tensor_config(info_dict)
        # self.dependencies = disambiguate_var_group(info_dict['dependencies'], SYMB)
        self.dependencies = info_dict['dependencies']
        self.array_name = self.name+self.array_name_complement
        self.gauss_name = self.name+self.gauss_name_complement

    def _latex(self, printer, exp=None, *args):
        if isinstance(printer, CustomLatexPrinter):
            # symbol, rank, dependencies = [printer._print(i) for i in self.args]
            if exp is None:
                return r"%s %s" % (self.latex_str, self.dependencies.get_tuple_string(printer))
            else:
                return r"%s^{%s} %s" % (self.latex_str, exp, self.dependencies.get_tuple_string(printer))
        else:
            return printer._print_Function(self, exp=exp)
        
    def substitute_gauss(self,expr):
        return self.substitute_components_simulatneous(expr, self.array, self.gauss)
        
class FunctionTensorPlaceholderRank0(FunctionTensorPlaceholder):
    rank = 0

    def generate_array(self, SYMB):
        dependencies_list = self.dependencies.get_dependency_list_array(SYMB,self.array_name_complement)
        self.array = sp.Function(self.array_name)(*dependencies_list)
        SYMB[self.array_name] = self.array

    def generate_gauss(self, SYMB):
        dependencies_list = self.dependencies.get_dependency_list_array_or_matrix(SYMB,self.gauss_name_complement)
        self.gauss = sp.Function(self.array_name)(*dependencies_list)
        SYMB[self.gauss_name] = self.gauss
    
    def substitute_gauss(self,expr):
        return expr.subs(self.array, self.gauss)

class FunctionTensorPlaceholderRank1(FunctionTensorPlaceholder):
    rank = 1

    def generate_array(self, SYMB):
        dependencies_list = self.dependencies.get_dependency_list_array(SYMB,self.array_name_complement)
        self.array = sp.Array([sp.Function(self.array_name+'_'+str(i))(*dependencies_list) for i in range(self.dim)])
        SYMB[self.array_name] = self.array
    
    def generate_gauss(self, SYMB):
        dependencies_list = self.dependencies.get_dependency_list_array_or_matrix(SYMB,self.gauss_name_complement)
        self.gauss = sp.Array([sp.Function(self.array_name+'_'+str(i))(*dependencies_list) for i in range(self.dim)])
        SYMB[self.gauss_name] = self.gauss

class FunctionTensorPlaceholderRank2(FunctionTensorPlaceholder):
    rank = 2

    def apply_tensor_config(self, info_dict):
        super().apply_tensor_config(info_dict)
        self.flag_symmetric = info_dict.get('symmetric',False)
        self.flag_voigt_notation = info_dict.get('use_voigt_notation',False)
    
    def generate_array(self, SYMB):
        dependencies_list = self.dependencies.get_dependency_list_array(SYMB,self.array_name_complement)
        self.array = self.fill_array(dependencies_list)
        SYMB[self.array_name] = self.array

    def generate_gauss(self, SYMB):
        dependencies_list = self.dependencies.get_dependency_list_array_or_matrix(SYMB,self.gauss_name_complement)
        self.gauss = self.fill_array(dependencies_list)
        SYMB[self.gauss_name] = self.gauss
    
    def fill_array(self, dependencies_list):
        out_array = sp.MutableDenseNDimArray(sp.zeros(self.dim**2),shape=(self.dim,self.dim))
        for i in range(self.dim):
            for j in range(self.dim):
                if self.flag_symmetric:
                    indexes_string = '_'+str(min(i,j))+'_'+str(max(i,j)) # Apply symmetry
                else:
                    indexes_string = '_'+str(i)+'_'+str(j)
                out_array[i,j] = sp.Function(self.array_name+indexes_string)(*dependencies_list)
        return out_array

class FunctionTensorPlaceholderRank4(FunctionTensorPlaceholder):
    rank = 4

    def apply_tensor_config(self, info_dict):
        super().apply_tensor_config(info_dict)
        self.flag_symmetric = info_dict.get('symmetric',False)
        if not self.flag_symmetric:
            raise NotImplementedError("Non-symmetric (on i-j and k-l pairs) 4th order tensors not implmented yet. Please set 'symmetric' to True")
        self.flag_third_symmetry = info_dict.get('third_symmetry',False)
        self.flag_voigt_notation = info_dict.get('use_voigt_notation',False)
    
    def generate_array(self, SYMB):
        dependencies_list = self.dependencies.get_dependency_list_array(SYMB,self.array_name_complement)
        self.array = self.fill_array(dependencies_list)
        SYMB[self.array_name] = self.array

    def generate_gauss(self, SYMB):
        dependencies_list = self.dependencies.get_dependency_list_array_or_matrix(SYMB,self.gauss_name_complement)
        self.gauss = self.fill_array(dependencies_list)
        SYMB[self.gauss_name] = self.gauss

    def fill_array(self, dependencies_list):
        out_array = sp.MutableDenseNDimArray(sp.zeros(self.dim**4),shape=(self.dim,self.dim,self.dim,self.dim))
        for i in range(self.dim):
            for j in range(self.dim):
                for k in range(self.dim):
                    for l in range(self.dim):
                        unique_index = [min(i,j), max(i,j), min(k,l), max(k,l)] # Apply first two symmetries
                        if self.flag_third_symmetry:
                            if unique_index[0] > unique_index[2]:
                                unique_index = [unique_index[2],unique_index[3],unique_index[0],unique_index[1]]
                            elif unique_index[0] == unique_index[2]:
                                if unique_index[1] > unique_index[3]:
                                    unique_index = [unique_index[2],unique_index[3],unique_index[0],unique_index[1]]
                        indexes_string=''
                        for index in unique_index:
                            indexes_string += '_'+str(index)
                        out_array[i,j,k,l] = sp.Function(self.array_name+indexes_string)(*dependencies_list)
        return out_array

class UnknownTensorPlaceholderRank0(FunctionTensorPlaceholderRank0):
    def apply_tensor_config(self, info_dict):
        super().apply_tensor_config(info_dict)
        self.N = info_dict['N']
        self.DN = info_dict['DN']
        self.deriv_array_name = self.name+'_deriv'+self.array_name_complement
        self.nodes_name = self.name + '_nodes'
        self.deriv_gauss_name = self.name+'_deriv'+self.gauss_name_complement

    def generate_array(self, SYMB):
        dependencies_list = self.dependencies.get_dependency_list_array(SYMB,'')
        self.array = sp.Function(self.array_name)(*dependencies_list)
        SYMB[self.array_name] = self.array
        self.deriv_array = sp.MutableDenseNDimArray(np.zeros((self.dim)))
        for i in range(self.dim):
            self.deriv_array[i] = sp.Derivative(self.array,dependencies_list[i])
        SYMB[self.deriv_array_name] = self.deriv_array

    def generate_gauss(self, SYMB):
        nnodes = self.N.shape[0]
        SYMB[self.nodes_name] = DefineVector(self.nodes_name,nnodes)
        self.gauss = SYMB[self.nodes_name].transpose()*self.N
        SYMB[self.gauss_name] = self.gauss
        self.deriv_gauss = DfjDxi(self.DN,SYMB[self.nodes_name])
        SYMB[self.deriv_gauss_name] = self.deriv_gauss

    def substitute_gauss(self, expr):
        # Substitute derivative first
        expr = self.substitute_components_simulatneous(expr, self.deriv_array, self.deriv_gauss)
        # Substitute unknown
        expr = expr.subs(self.array, self.gauss[0,0])
        return expr

class UnknownTensorPlaceholderRank1(FunctionTensorPlaceholderRank1):
    def apply_tensor_config(self, info_dict):
        super().apply_tensor_config(info_dict)
        self.N = info_dict['N']
        self.DN = info_dict['DN']
        self.deriv_array_name = self.name+'_deriv'+self.array_name_complement
        self.nodes_name = self.name + '_nodes'
        self.deriv_gauss_name = self.name+'_deriv'+self.gauss_name_complement

    def generate_array(self, SYMB):
        dependencies_list = self.dependencies.get_dependency_list_array(SYMB,'')
        self.array = sp.Array([sp.Function(self.array_name+'_'+str(i))(*dependencies_list) for i in range(self.dim)])
        SYMB[self.array_name] = self.array
        self.deriv_array = sp.MutableDenseNDimArray(np.zeros((self.dim,self.dim)))
        for i in range(self.dim):
            for j in range(self.dim):
                self.deriv_array[i,j] = sp.Derivative(self.array[j],dependencies_list[i])
        SYMB[self.deriv_array_name] = self.deriv_array
    
    def generate_gauss(self, SYMB):
        nnodes = self.N.shape[0]
        SYMB[self.nodes_name] = DefineMatrix(self.nodes_name,nnodes,self.dim)
        self.gauss = SYMB[self.nodes_name].transpose()*self.N
        SYMB[self.gauss_name] = self.gauss
        self.deriv_gauss = DfjDxi(self.DN,SYMB[self.nodes_name])
        SYMB[self.deriv_gauss_name] = self.deriv_gauss

    def substitute_gauss(self, expr):
        # Substitute derivative first
        expr = self.substitute_components_simulatneous(expr, self.deriv_array, self.deriv_gauss)
        # Substitute unknown
        expr = self.substitute_components_simulatneous(expr, self.array, self.gauss)
        return expr

class SymbolicTensorPlaceholder(BaseTensorPlaceholder):
    array_name_complement = '_array'
    gauss_name_complement = '_gauss'

    def apply_tensor_config(self, info_dict):
        super().apply_tensor_config(info_dict)
        self.array_name = self.name+self.array_name_complement
        self.gauss_name = self.name+self.gauss_name_complement
        self.flag_positive = info_dict.get('positive',None)
        self.strain_size  = self.dim*(self.dim+1)//2
        
class SymbolicTensorPlaceholderRank0(SymbolicTensorPlaceholder):
    rank = 0

    def generate_array(self, SYMB):
        self.array = sp.Symbol(self.array_name, positive=self.flag_positive)
        SYMB[self.array_name] = self.array

    def generate_gauss(self, SYMB):
        self.gauss = sp.Symbol(self.gauss_name, positive=self.flag_positive)
        SYMB[self.gauss_name] = self.gauss

    def substitute_gauss(self, expr):
        return expr.subs(self.array, self.gauss)

class SymbolicTensorPlaceholderRank1(SymbolicTensorPlaceholder):
    rank = 1

    def generate_array(self, SYMB):
        self.array = sp.Array([sp.Symbol(self.array_name+'_'+str(i), positive=self.flag_positive) for i in range(self.dim)])
        SYMB[self.array_name] = self.array

    def generate_gauss(self, SYMB):
        self.gauss = DefineVector(self.gauss_name, self.dim)
        SYMB[self.gauss_name] = self.gauss

    def substitute_gauss(self, expr):
        return self.substitute_components_simulatneous(expr, self.array, self.gauss)

class SymbolicTensorPlaceholderRank2(SymbolicTensorPlaceholder):
    rank = 2

    def apply_tensor_config(self, info_dict):
        super().apply_tensor_config(info_dict)
        self.flag_symmetric = info_dict.get('symmetric',False)
        self.flag_voigt_notation = info_dict.get('use_voigt_notation',False)

    def generate_array(self, SYMB):
        self.array = sp.MutableDenseNDimArray(sp.zeros(self.dim**2),shape=(self.dim,self.dim))
        for i in range(self.dim):
            for j in range(self.dim):
                if self.flag_symmetric:
                    indexes_string = '_'+str(min(i,j))+'_'+str(max(i,j)) # Apply symmetry
                else:
                    indexes_string = '_'+str(i)+'_'+str(j)
                self.array[i,j] = sp.Symbol(self.array_name+indexes_string, positive=self.flag_positive)
        SYMB[self.array_name] = self.array

    def generate_gauss(self, SYMB):
        if self.flag_voigt_notation:
            assert self.flag_symmetric
            self.gauss = DefineVector(self.gauss_name,self.strain_size)
        elif self.flag_symmetric:
            self.gauss = DefineSymmetricMatrix(self.gauss_name,self.dim,self.dim)
        else:
            self.gauss = DefineMatrix(self.gauss_name,self.dim,self.dim)
        SYMB[self.gauss_name] = self.gauss

    def substitute_gauss(self, expr):
        if self.flag_voigt_notation:
            for i in range(self.strain_size):
                idx_list = VOIGT_INDEX_DICT[self.dim][i]
                expr = expr.subs(self.array[idx_list],self.gauss[i])
        else:
            expr = self.substitute_components_simulatneous(expr, self.array, self.gauss)
        return expr

class SymbolicTensorPlaceholderRank4(SymbolicTensorPlaceholder):
    rank = 4

    def apply_tensor_config(self, info_dict):
        super().apply_tensor_config(info_dict)
        self.flag_symmetric = info_dict.get('symmetric',False)
        if not self.flag_symmetric:
            raise NotImplementedError("Non-symmetric (on i-j and k-l pairs) 4th order tensors not implmented yet. Please set 'symmetric' to True")
        self.flag_third_symmetry = info_dict.get('third_symmetry',False)
        self.flag_voigt_notation = info_dict.get('use_voigt_notation',False)
    
    def generate_array(self, SYMB):
        self.array = sp.MutableDenseNDimArray(sp.zeros(self.dim**4),shape=(self.dim,self.dim,self.dim,self.dim))
        for i in range(self.dim):
            for j in range(self.dim):
                for k in range(self.dim):
                    for l in range(self.dim):
                        unique_index = [min(i,j), max(i,j), min(k,l), max(k,l)] # Apply first two symmetries
                        if self.flag_third_symmetry:
                            if unique_index[0] > unique_index[2]:
                                unique_index = [unique_index[2],unique_index[3],unique_index[0],unique_index[1]]
                            elif unique_index[0] == unique_index[2]:
                                if unique_index[1] > unique_index[3]:
                                    unique_index = [unique_index[2],unique_index[3],unique_index[0],unique_index[1]]
                        indexes_string=''
                        for index in unique_index:
                            indexes_string += '_'+str(index)
                        self.array[i,j,k,l] = sp.Symbol(self.array_name+indexes_string, positive=self.flag_positive)
        SYMB[self.array_name] = self.array

    def generate_gauss(self, SYMB):
        if self.flag_voigt_notation:
            assert self.flag_symmetric and self.flag_third_symmetry
            self.gauss = DefineSymmetricMatrix(self.gauss_name,self.strain_size,self.strain_size)
        else:
            raise NotImplementedError("Only voigt notation availabe for tensors of rank 4")
        SYMB[self.gauss_name] = self.gauss

    def substitute_gauss(self, expr):
        if self.flag_voigt_notation:
            for i in range(self.strain_size):
                for j in range(self.strain_size):
                    idx_list = [*VOIGT_INDEX_DICT[self.dim][i],*VOIGT_INDEX_DICT[self.dim][j]]
                    expr = expr.subs(self.array[idx_list],self.gauss[i,j])
        else:
            raise NotImplementedError("Only voigt notation availabe for tensors of rank 4")
        return expr

class NumericalTensorPlaceholder(BaseTensorPlaceholder):
    array_name_complement = '_numerical'

    def apply_tensor_config(self, info_dict):
        super().apply_tensor_config(info_dict)
        self.array_name = self.name+self.array_name_complement
        self.rank = info_dict['tensor_rank']
        self.value = info_dict['value']
        
    def generate_array(self, SYMB):
        self.array = sp.Array(self.value)
        SYMB[self.array_name] = self.array

    def generate_gauss(self, SYMB):
        self.gauss = None

    def substitute_gauss(self, expr):
        return expr
    
class NumericalTensorPlaceholderRank0(NumericalTensorPlaceholder):

    def generate_array(self, SYMB):
        self.array = sp.sympify(self.value)
        SYMB[self.array_name] = self.array
    
class DefinedFunctionPlaceholder(BaseTensorPlaceholder):
    array_name_complement = '_defined_function'

    def apply_tensor_config(self, info_dict):
        super().apply_tensor_config(info_dict)
        self.array_name = self.name+self.array_name_complement
        self.rank = 0
        self.value = info_dict['value']
        
    def generate_explicit_functions(self):
        substituted_expr = substitute_symbols(self.value)
        substituted_expr = substitute_functions(substituted_expr)
        return substituted_expr
        
    def generate_array(self, SYMB):
        pass

    def generate_gauss(self, SYMB):
        pass

    def substitute_gauss(self, expr):
        pass