import numpy as np
import sympy as sp
import ast

from lib.basic_classes import BaseTensorPlaceholder, VarsCombination, CoefficientsIndicator, CoordsIndicator
from lib.array_definitions import DefineVector, DefineMatrix, DefineSymmetricMatrix
from lib.kratos_utilities import DfjDxi
from lib.printers import CustomLatexPrinter

from lib.coordinates_system import ACTIVE_COORD_SYSTEM

VOIGT_INDEX_DICT = {
    2: [[0,0],[1,1],[0,1]],
    # 3: [[0,0],[1,1],[2,2],[1,2],[0,2],[0,1]] # Standard Voigt notation
    3: [[0,0],[1,1],[2,2],[0,1],[1,2],[0,2]]  # Kratos Voigt notation
    }
    
class NodalTensorPlaceholder(BaseTensorPlaceholder):
    nodes_name_complement = '_nodes'

    @classmethod
    def from_info_dict(cls, info_dict):
        if info_dict["tensor_rank"] >= 2:
            raise NotImplementedError("Only nodal functions of tensor rank 0 and 1 are supported.")
        
        args_list = super().parse_info_dict(info_dict)

        element_space_name = sp.core.symbol.Str(info_dict['element_space_name'])
        args_list.append(element_space_name)

        return cls(*args_list)
    
    @property
    def element_space_name(self):
        return str(self.args[6])
    
    @property
    def deriv_array(self):
        return sp.derive_by_array(self.array, self.dependencies)
    
    @property
    def nodes(self):
        N = ACTIVE_COORD_SYSTEM.get()["element_spaces_dict"][self.element_space_name]["N"]
        nnodes = N.shape[0]
        nodes_name = self.name+self.nodes_name_complement
        if self.rank==0:
            return DefineVector(nodes_name,nnodes)
        elif self.rank==1:
            return DefineMatrix(nodes_name,nnodes,self.dim[0])
    
    @property
    def gauss(self):
        N = ACTIVE_COORD_SYSTEM.get()["element_spaces_dict"][self.element_space_name]["N"]
        return self.nodes.transpose()*N
    
    @property
    def deriv_gauss(self):
        DN = ACTIVE_COORD_SYSTEM.get()["element_spaces_dict"][self.element_space_name]["DN"]
        return DfjDxi(DN,self.nodes)
    
    def substitute_arrays_to_gauss(self,expr):
        # Substitute derivative first
        expr = self.substitute_components_simulatneous(expr, self.deriv_array, self.deriv_gauss)
        # Substitute unknown
        if self.rank==0:
            expr = expr.subs(self.array, self.gauss[0,0])
        elif self.rank==1:
            expr = self.substitute_components_simulatneous(expr, self.array, self.gauss)
        return expr
    
class UnknownTensorPlaceholder(NodalTensorPlaceholder):
    pass

class SymbolicTensorPlaceholder(BaseTensorPlaceholder):

    @property
    def strain_size(self):
        if self.rank in [2,4] and len(set(self.dim))==1:
            return self.dim[0]*(self.dim[0]+1)//2
        else:
            raise ValueError("Invalid tensor rank or dimensions for obtaining strain_size")

    @property
    def gauss(self):
        flags = [str(flag) for flag in self.flags] # Convert to strings for comparison
        gauss_name = self.name+self.gauss_name_complement
        if self.rank == 0:
            positive_flag = "positive" in flags
            gauss = sp.Symbol(gauss_name, positive=positive_flag)
        elif self.rank == 1:
            gauss =  DefineVector(gauss_name, self.dim[0])
        elif self.rank == 2:
            if "symmetric" in flags:
                gauss = DefineSymmetricMatrix(gauss_name, self.dim[0], self.dim[1])
                if "use_voigt_notation" in flags:
                    gauss = DefineVector(gauss_name,self.strain_size)
            else:
                gauss = DefineMatrix(gauss_name, self.dim[0], self.dim[1])
        elif self.rank == 4:
            if "use_voigt_notation" in flags:
                assert "symmetric" in flags and "third_symmetry" in flags
                gauss = DefineSymmetricMatrix(gauss_name,self.strain_size,self.strain_size)
            else:
                raise NotImplementedError("Only voigt notation availabe for tensors of rank 4")
        else:
            raise NotImplementedError("Invalid tensor rank")
        return gauss
            

    def substitute_arrays_to_gauss(self,expr):
        flags = [str(flag) for flag in self.flags] # Convert to strings for comparison
        if self.rank==0:
            expr = expr.subs(self.array, self.gauss)
        elif self.rank == 2 and "use_voigt_notation" in flags:
            for i in range(self.strain_size):
                idx_list = VOIGT_INDEX_DICT[self.dim[0]][i]
                expr = expr.subs(self.array[idx_list],self.gauss[i])
        elif self.rank == 4 and "use_voigt_notation" in flags:
            for i in range(self.strain_size):
                for j in range(self.strain_size):
                    idx_list = [*VOIGT_INDEX_DICT[self.dim[0]][i],*VOIGT_INDEX_DICT[self.dim[0]][j]]
                    expr = expr.subs(self.array[idx_list],self.gauss[i,j])
        else:
            expr = self.substitute_components_simulatneous(expr, self.array, self.gauss)
        return expr
    
class ConstantTensorPlaceholder(SymbolicTensorPlaceholder):
    # It behaves just like a SymbolicTensorPlaceholder but there will be no dependencies
    
    @property
    def array(self):
        array_name = self.name+self.array_name_complement
        return self._generate_functions_array(array_name, None)

class UndefinedFunctionTensorPlaceholder(BaseTensorPlaceholder):
    
    @classmethod
    def from_info_dict(cls, info_dict, namespace):
        dependencies = info_dict.get("dependencies", None)
        if isinstance(dependencies, str):
            if isinstance(namespace[dependencies], CoefficientsIndicator) or isinstance(namespace[dependencies], CoordsIndicator):
                dependencies = namespace[dependencies]
            else:
                dependencies = [dependencies]
        if isinstance(dependencies, list):
            dependencies = VarsCombination.from_namespace(dependencies, namespace)
        info_dict["dependencies"] = dependencies
        return super().from_info_dict(info_dict)

    def _latex(self, printer, exp=None, *args):
        if isinstance(printer, CustomLatexPrinter):
            # symbol, rank, dependencies = [printer._print(i) for i in self.args]
            if exp is None:
                return r"%s%s" % (self.latex_str, self.dependencies.get_tuple_string(printer))
            else:
                return r"%s^{%s}%s" % (self.latex_str, exp, self.dependencies.get_tuple_string(printer))
        else:
            return printer._print_Function(self, exp=exp)
        
    @property
    def array(self):
        array_name = self.name+self.array_name_complement
        dependencies = self.dependencies
        if isinstance(dependencies, VarsCombination):
            dependencies = dependencies.array
        return self._generate_functions_array(array_name, dependencies)
    
    @property
    def gauss(self):
        gauss_dependencies_list = self.dependencies.gauss
        # Then we generate the function or array of functions
        gauss_name = self.name+self.gauss_name_complement
        return self._generate_functions_array(gauss_name, gauss_dependencies_list)
    
    def substitute_arrays_to_gauss(self,expr):
        expr = self.substitute_components_simulatneous(expr, self.array, self.gauss)
        return expr

class NumericalTensorPlaceholder(BaseTensorPlaceholder):

    @property
    def array(self):
        
        value_str = str(self.dependencies).strip()

        if value_str == "identity": value = np.eye(self.dim[0], dtype=int)
        elif value_str == "zeros": value = np.zeros(self.dim, dtype=int)
        elif value_str == "ones": value = np.ones(self.dim, dtype=int)
        else:
            # Try to parse as a number or list
            value = ast.literal_eval(value_str)

        if self.rank==0:
            return value
        else:
            array = sp.Array(value)
            assert array.shape == tuple(self.dim)
            return array


    @property
    def gauss(self):
        return None
    
    def substitute_arrays_to_gauss(self,expr):
        return expr
    
class DefinedFunctionTensorPlaceholder(BaseTensorPlaceholder):
    # The dependencies should contain the expression defining the function
    # After generating the functional expression in compact form and printing latex,
    # the compiler will substitute these placeholders by their actual definition

    @property
    def array(self):
        return None

    @property
    def gauss(self):
        return None