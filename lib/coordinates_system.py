import contextvars

from lib.kratos_utilities import DefineShapeFunctions

ACTIVE_COORD_SYSTEM = contextvars.ContextVar("ACTIVE_COORD_SYSTEM", default=None)

class CoordinateSystem():
    def __init__(self, coord_symbols, nnodes_dict, impose_partion_of_unity, transposed_gradients_flag=False):
        
        element_spaces_dict = dict()
        for name in nnodes_dict.keys():
            if len(nnodes_dict.keys()) == 1:
                N,DN = DefineShapeFunctions(nnodes_dict[name], len(coord_symbols), impose_partion_of_unity)
            else:
                N,DN = DefineShapeFunctions(nnodes_dict[name], len(coord_symbols), impose_partion_of_unity, shape_functions_name=f'N_{name}', first_derivatives_name=f'DN_{name}')
            element_spaces_dict[name]=dict()
            element_spaces_dict[name]["N"] = N
            element_spaces_dict[name]["DN"] = DN
        
        self.coord_system = {
            "coord_symbols": coord_symbols,
            "element_spaces_dict": element_spaces_dict,
            "transposed_gradients_flag": transposed_gradients_flag
        }
        self.coord_token = None
        

    def __enter__(self):
        # Set the active system for the duration of the 'with' block
        self.coord_token = ACTIVE_COORD_SYSTEM.set(self.coord_system)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Reset to the previous state (or None)
        ACTIVE_COORD_SYSTEM.reset(self.coord_token)