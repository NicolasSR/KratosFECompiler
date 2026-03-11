import importlib
import json
from pathlib import Path

import cppyy
from docopt import docopt
import numpy as np

from kratos_fe_compiler.metadata import VERSION
from lib.logger import main_logger


PROGRAM_NAME = "numerical_cpp_compare"
HELP = """Comparing Tool for FE C++ files. Compares numerically by assigning random values

Usage:
    {p} [options] CASE_NAME

Options:
    --source=DIR        Dir for the cases specification [default: ./cases]

    -D, --debug         Show debug messages
    -q, --quiet         Show less messages

    -h, --help          Show this screen and exit
    -V, --version       Show version
""".format(
    p = PROGRAM_NAME
)

class NumericCPPComparingEngine():

    def __init__(self, source_dir: Path, case_name: str):
        self.case_name = case_name
        self.case_output_dir=source_dir/self.case_name/"output"

        with open(self.case_output_dir/"cpp_interface.json", "r") as f:
            cpp_interface_info = json.load(f)

        self.dim = cpp_interface_info["dim"]
        self.nnodes = cpp_interface_info["nnodes"]
        self.num_dofs = cpp_interface_info["dofs"]
        self.vars_dict = cpp_interface_info["vars"]

        self.gen_shapes_map()

        self._compiled_auto = False
        self._compiled_manual = False

        self.rng = np.random.default_rng(1793)

    def gen_shapes_map(self):
        sym_unique_size = self.dim*(self.dim+1)/2
        self.shape_map = {
            'scalars': [1],
            'nodal_scalars': [self.nnodes],
            'vectors': [self.dim],
            'nodal_vectors': [self.nnodes, self.dim],
            'sym_matrices': [sym_unique_size],
            'matrices': [self.dim,self.dim],
            'rank_4_voigt': [sym_unique_size,sym_unique_size]
        }

    def _compile_auto(self, vars_dict):
        """Compiles the C++ code exactly once."""
        if self._compiled_auto:
            return
        
        self._compile("NS_AUTO", self.case_output_dir / "RHS.cpp", self.case_output_dir / "LHS.cpp", vars_dict)
        
        self._compiled_auto = True

    def _compile_manual(self, vars_dict):
        """Compiles the C++ code exactly once."""
        if self._compiled_manual:
            return
        
        self._compile("NS_MANUAL", self.case_output_dir / "manual_RHS.cpp", self.case_output_dir / "manual_LHS.cpp", vars_dict)
        
        self._compiled_manual = True

    def _compile(self, namespace_name, rhs_path, lhs_path, vars_dict):
        # Build the signature dynamically
        arg_list = ["double* rRightHandSideVector", "double* rLeftHandSideMatrix_ptr", "double w_g", "double* N", "double* DN_ptr"]
        
        # Local references inside the function
        mappings = []
        undef_mappings = []

        for var_type, vars_list in vars_dict.items():
            for var in vars_list:
                if var_type == "scalars":
                    arg_list.append(f"double {var}")
                else:
                    shape = self.shape_map[var_type]
                    if len(shape) == 2:
                        mappings.append(f"#define {var}(i, j) {var}_ptr[(i)*{int(shape[1])}+(j)]")
                        undef_mappings.append(f"#undef {var}")
                        arg_list.append(f"double* {var}_ptr")
                    else:
                        arg_list.append(f"double* {var}")

        with open(rhs_path, "r") as f:
            rhs_body = f.read()
        with open(lhs_path, "r") as f:
            lhs_body = f.read()

        full_code = f"""
        namespace {namespace_name} {{
            void evaluate_snippet({', '.join(arg_list)}) {{
                #define rLeftHandSideMatrix(i, j) rLeftHandSideMatrix_ptr[(i)*{self.num_dofs}+(j)]
                #define DN(i, j) DN_ptr[(i)*{self.dim}+(j)]
                {chr(10).join(mappings)}
                
                {rhs_body}
                {lhs_body}

                //Undef to avoid leaking to other snippets
                #undef DN
                {chr(10).join(undef_mappings)}
            }}
        }}
        """
        cppyy.cppdef(full_code)

    def make_scalar(array):
        return array[0]

    def flatten(matrix):
        return matrix.flatten()
    
    def make_symmetric_and_flatten(matrix):
        sym_matrix = (matrix+matrix.T)/2
        return sym_matrix.flatten()
    
    shape_post_processing_map = {
            'scalars': make_scalar,
            'nodal_vectors': flatten,
            'rank_4_voigt': make_symmetric_and_flatten
        }

    def generate_values_set(self):
        values_dict = {
            "w_g": self.rng.random(),
            "N": self.rng.random((self.nnodes)),
            "DN": self.rng.random((self.nnodes, self.dim)).flatten()
        }
        for var_type, vars_list in self.vars_dict.items():
            for var in vars_list:
                values = self.rng.random([int(s) for s in self.shape_map[var_type]])
                if var_type in self.shape_post_processing_map.keys():
                    values = self.shape_post_processing_map[var_type](values)
                values_dict[var] = values

        return values_dict
    

    def compute_auto(self, values_dict):
        self._compile_auto(self.vars_dict)

        # Ensure the result buffer is contiguous and correct type
        result_rhs_auto = np.zeros(self.num_dofs, dtype=np.float64)
        result_lhs_auto = np.zeros(self.num_dofs**2, dtype=np.float64)
        
        # Extract values in the SAME order as the C++ signature
        # IMPORTANT: Dictionary order is only guaranteed in Python 3.7+
        # It is safer to iterate through your vars_dict keys explicitly
        call_args = [values_dict["w_g"], values_dict["N"], values_dict["DN"]]
        
        for var_type, vars_list in self.vars_dict.items():
            for var in vars_list:
                call_args.append(values_dict[var])

        # cppyy handles the conversion from numpy array to double* automatically
        cppyy.gbl.NS_AUTO.evaluate_snippet(result_rhs_auto, result_lhs_auto, *call_args)
        return result_rhs_auto, result_lhs_auto
    
    def compute_manual(self, values_dict):
        self._compile_manual(self.vars_dict)

        # Ensure the result buffer is contiguous and correct type
        result_rhs_manual = np.zeros(self.num_dofs, dtype=np.float64)
        result_lhs_manual = np.zeros(self.num_dofs**2, dtype=np.float64)
        
        # Extract values in the SAME order as the C++ signature
        # IMPORTANT: Dictionary order is only guaranteed in Python 3.7+
        # It is safer to iterate through your vars_dict keys explicitly
        call_args = [values_dict["w_g"], values_dict["N"], values_dict["DN"]]
        
        for var_type, vars_list in self.vars_dict.items():
            for var in vars_list:
                call_args.append(values_dict[var])

        # cppyy handles the conversion from numpy array to double* automatically
        cppyy.gbl.NS_MANUAL.evaluate_snippet(result_rhs_manual, result_lhs_manual, *call_args)
        return result_rhs_manual, result_lhs_manual
    
    def print_max_diff_n_times(self, n=100):
        max_rhs = 0
        max_lhs = 0
        for i in range(n):
            values_dict = self.generate_values_set()
            cresult_rhs_auto, result_lhs_auto = self.compute_auto(values_dict)
            result_rhs_manual, result_lhs_manual = self.compute_manual(values_dict)

            diff_rhs = cresult_rhs_auto-result_rhs_manual
            diff_lhs = result_lhs_auto-result_lhs_manual

            max_rhs = max(max_rhs,np.max(np.abs(diff_rhs)))
            max_lhs = max(max_lhs,np.max(np.abs(diff_lhs)))

        print('Max diff RHS:', max_rhs)
        print('Max diff LHS:', max_lhs)


if __name__ == "__main__":
    # parse options
    args = docopt(HELP, version=VERSION)

    source_dir = Path(args['--source'])
    case_name = args['CASE_NAME']

    main_logger.set_logger(args['--quiet'], args['--debug'])

    comparing_engine = NumericCPPComparingEngine(source_dir, case_name)
    comparing_engine.print_max_diff_n_times()