import unittest

import sympy as sp

from lib.tensor_placeholders import *
from lib.basic_classes import CoefficientsIndicator, CoordsIndicator
from lib.coordinates_system import CoordinateSystem


class TestDofsIndicator(unittest.TestCase):

    def test_1(self):
        
        dim = 2
        nnodes = 3
        impose_partion_of_unity = False
        transpose_gradients_flag = False

        namespace = {}

        namespace['x'] = sp.Symbol('x')
        x = namespace['x']
        namespace['y'] = sp.Symbol('y')
        y = namespace['y']
        base_scalars = CoordsIndicator.from_namespace(['x','y'],namespace)

        self.assertEqual(base_scalars, CoordsIndicator(x,y), "CoordsIndicator not correctly defined from namespace")

        self.assertEqual(base_scalars.array, [x,y], "CoordsIndicator array not correctly defined")

        namespace["v"] = UnknownTensorPlaceholder.from_info_dict({'symbol': 'v', 'latex':'v',
                                    'tensor_rank': 1, 'dim':[dim], 'dependencies': base_scalars})
        namespace["p"] = UnknownTensorPlaceholder.from_info_dict({'symbol': 'p', 'latex':'p',
                                    'tensor_rank': 0, 'dim':[], 'dependencies': base_scalars})
        
        # Check that v and p's dependencies are correctly set
        self.assertEqual(namespace["v"].dependencies, base_scalars, "v's dependencies not correctly set")
        self.assertEqual(namespace["p"].dependencies, base_scalars, "p's dependencies not correctly set")

        manual_v_array = sp.Array([sp.Function("v_array_0")(x,y),sp.Function("v_array_1")(x,y)])
        manual_p_array = sp.Function("p_array")(x,y)

        self.assertEqual(namespace["v"].array, manual_v_array, "v's array not correctly defined")
        self.assertEqual(namespace["p"].array, manual_p_array, "p's array not correctly defined")
        
        trial_symbols_list = ["v","p"]
        dofs = CoefficientsIndicator.from_namespace(trial_symbols_list, namespace)
        namespace['dofs'] = dofs

        self.assertEqual(namespace["dofs"], CoefficientsIndicator(namespace["v"],namespace["p"]), "dofs not correctly defined from namespace")
        
        self.assertEqual(namespace["dofs"].array, namespace["dofs"])

        namespace["E"] = UndefinedFunctionTensorPlaceholder.from_info_dict({'symbol': 'E', 'latex':'E',
                                'tensor_rank': 2, 'dim':[2,2], 'dependencies': 'dofs'}, namespace)
        
        self.assertEqual(namespace["E"].dependencies, dofs, "E's dependencies not correctly set")

        E_array_manual = sp.Array([[sp.Function("E_array_0_0")(*dofs.array),sp.Function("E_array_0_1")(*dofs.array)],
                    [sp.Function("E_array_1_0")(*dofs.array),sp.Function("E_array_1_1")(*dofs.array)]])
        self.assertEqual(namespace["E"].array, E_array_manual)

        material_coords_system = CoordinateSystem(base_scalars, nnodes, impose_partion_of_unity, transpose_gradients_flag)
        
        # We apply our coordinate system
        with material_coords_system:

            dofs_gauss_manual = [namespace["v"].nodes[0,0],namespace["v"].nodes[0,1],namespace["p"].nodes[0],
                                 namespace["v"].nodes[1,0],namespace["v"].nodes[1,1],namespace["p"].nodes[1],
                                 namespace["v"].nodes[2,0],namespace["v"].nodes[2,1],namespace["p"].nodes[2]] 
            self.assertEqual(namespace["dofs"].gauss, dofs_gauss_manual)

if __name__ == "__main__":
    unittest.main()