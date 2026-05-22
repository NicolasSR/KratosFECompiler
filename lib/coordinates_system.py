import contextvars

from lib.kratos_utilities import DefineShapeFunctions

ACTIVE_COORD_SYSTEM = contextvars.ContextVar("ACTIVE_COORD_SYSTEM", default=None)

class CoordinateSystem():
    def __init__(self, coord_symbols, nnodes, impose_partion_of_unity, transposed_gradients_flag=False):
        N,DN = DefineShapeFunctions(nnodes, len(coord_symbols), impose_partion_of_unity)
        self.coord_system = {
            "coord_symbols": coord_symbols,
            "N": N,
            "DN": DN,
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