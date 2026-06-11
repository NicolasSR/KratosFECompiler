import unittest
from itertools import permutations

import sympy as sp

from lib.tensor_placeholders import *
from lib.basic_classes import VarsCombination, CoordsIndicator
import lib.placeholder_operators as po


from lib.substitution_utils import substitute_all_placeholders_in_expression
from lib.coordinates_system import CoordinateSystem


class TestVarsCombination(unittest.TestCase):

    def test_1(self):
        nnodes = 3
        dim = 2
        impose_partion_of_unity = False
        transpose_gradients_flag = False

        namespace = {}

        namespace['x'] = sp.Symbol('x')
        x = namespace['x']
        namespace['y'] = sp.Symbol('y')
        y = namespace['y']
        base_scalars = CoordsIndicator.from_namespace(['x','y'],namespace)

        # Assign new TensorPlaceholders to all variables
        namespace['B'] = SymbolicTensorPlaceholder.from_info_dict({'symbol': 'B', 'latex':'B',
                                'tensor_rank': 2, 'dim':[dim]*2, 'dependencies': base_scalars,
                                'symmetric': False, 'third_symmetry': False,
                                'use_voigt_notation': False, "positive": False})
        namespace['c'] = SymbolicTensorPlaceholder.from_info_dict({'symbol': 'c', 'latex':'c',
                                'tensor_rank': 1, 'dim':[dim], 'dependencies': base_scalars,
                                'symmetric': False, 'third_symmetry': False,
                                'use_voigt_notation': False, "positive": False})
        namespace['rho'] = SymbolicTensorPlaceholder.from_info_dict({'symbol': 'rho', 'latex':'\\rho',
                                'tensor_rank': 0, 'dim':[], 'dependencies': base_scalars,
                                'symmetric': False, 'third_symmetry': False,
                                'use_voigt_notation': False, "positive": False})
        namespace['A'] = UndefinedFunctionTensorPlaceholder.from_info_dict({'symbol': 'A', 'latex':'A',
                                'tensor_rank': 2, 'dim':[dim]*2, 'dependencies': ['B','c','rho']}, namespace)
        namespace['d'] = UndefinedFunctionTensorPlaceholder.from_info_dict({'symbol': 'd', 'latex':'d',
                                'tensor_rank': 0, 'dim':[], 'dependencies': 'A'}, namespace)
        
        # Check that A's dependencies are correctly set

        vars_comb = VarsCombination.from_namespace(['B','c','rho'], namespace)

        manual_vars_comb_array = [sp.Function("B_array_0_0")(x,y),sp.Function("B_array_0_1")(x,y),
                    sp.Function("B_array_1_0")(x,y),sp.Function("B_array_1_1")(x,y),
                    sp.Function("c_array_0")(x,y),sp.Function("c_array_1")(x,y),
                    sp.Function("rho_array")(x,y)]

        self.assertEqual(vars_comb, VarsCombination(namespace['B'], namespace['c'], namespace['rho']), "VarsCombination not correctly created from namespace")

        self.assertEqual(vars_comb.array, manual_vars_comb_array, "VarsCombination array not correctly defined")

        self.assertEqual(namespace['A'].dependencies, vars_comb, "A's dependencies not correctly set")

        self.assertEqual(namespace['A'].dependencies.array, manual_vars_comb_array, "A's dependencies array not correctly defined")

        manual_A_array = sp.Array([[sp.Function("A_array_0_0")(*vars_comb.array),sp.Function("A_array_0_1")(*vars_comb.array)],
                    [sp.Function("A_array_1_0")(*vars_comb.array),sp.Function("A_array_1_1")(*vars_comb.array)]])
        
        self.assertEqual(namespace['A'].array, manual_A_array, "A's array not correctly defined")

        flattened_A_array = [manual_A_array[0,0],manual_A_array[0,1],manual_A_array[1,0],manual_A_array[1,1]]

        self.assertEqual(namespace['d'].array, sp.Function("d_array")(*flattened_A_array), "d's array not correctly defined")

        # Define coordinates system (this will generate matrix for shape functions and their derivatives)
        material_coords_system = CoordinateSystem(base_scalars, nnodes, impose_partion_of_unity, transpose_gradients_flag)

        
        # We apply our coordinate system
        with material_coords_system:

            # Code for the functional is built from SSA (Static Single Assignment)-based AST in JSON format

            namespace["functional_rhs"] = po.matrix_det_op(namespace['B']+namespace['A'])+namespace['d']
            functional_rhs = namespace["functional_rhs"]

            # If we substitute the B for its array, it should not be substituted within A's dependencies
            func_array_0 = substitute_all_placeholders_in_expression(functional_rhs, ["B"], namespace)
            self.assertEqual(func_array_0.args[0].args[0].args[1].dependencies, vars_comb, "Substitution of B's array affected A's dependencies, which should not happen")
            self.assertEqual(func_array_0.args[1].dependencies, VarsCombination.from_namespace("A", namespace), "Substitution of B's array affected d's dependencies, which should not happen")
            func_array_1 = substitute_all_placeholders_in_expression(func_array_0, ["A"], namespace)
            self.assertEqual(func_array_1.args[1].dependencies, VarsCombination.from_namespace("A", namespace), "Substitution of A's array affected d's dependencies, which should not happen")
            

            tensor_names = {"B","c","A","d","rho"}
            for i, p in enumerate(permutations(tensor_names)):
                if i == 0:
                    ref = substitute_all_placeholders_in_expression(functional_rhs, list(p), namespace)
                else:
                    self.assertEqual(substitute_all_placeholders_in_expression(functional_rhs, list(p), namespace), ref, "Array substitution with VarsCombination is sensitive to substitution order")


if __name__ == "__main__":
    unittest.main()