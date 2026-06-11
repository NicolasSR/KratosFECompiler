import importlib
from pathlib import Path
import json

from docopt import docopt

from kratos_fe_compiler.metadata import VERSION
from kratos_fe_compiler.compiler import KratosFECompiler
from lib.logger import main_logger


PROGRAM_NAME = "kratos_fe_compiler"
HELP = """Sympy-based compiler for Finite Elements within KratosMultiphysics

Usage:
    {p} [options] CASE_NAME

Options:
    --overwrite         Overwrite output files if already exist

    --type=ARG          Type of generation to use ('fe_compiler', 'manual', 'all') [default: fe_compiler]

    --source=DIR        Dir for the cases specification [default: ./cases]

    -D, --debug         Show debug messages
    -q, --quiet         Show less messages

    -h, --help          Show this screen and exit
    -V, --version       Show version
""".format(
    p = PROGRAM_NAME
)

class SymbolicGenerator():

    def __init__(self, source_dir: Path, case_name: str, overwrite: bool):
        self.case_name = case_name
        self.case_dir = source_dir / self.case_name
        self.case_output_dir = self.case_dir / "output"
        self.case_import_root = f"{source_dir}.{case_name}"
        self.overwrite = overwrite
        self.get_case_config()

    def get_case_config(self):
        with open(self.case_dir/"case_config.json", 'r') as f:
            case_config = json.load(f)
        self.applied_config = case_config["applied_configuration"]
        ## Check that dim and number of nodes are compatible:
        dim = str(self.applied_config["dim"])
        nnodes = self.applied_config["nnodes"]
        ## Check that formulation type, dim and number of nodes are compatible:
        if (not dim in case_config["compatibilities_dict"].keys()) or (not nnodes in case_config["compatibilities_dict"][dim]):
            err_msg = "Wrong Dimensions or Number of Nodes"
            raise Exception(err_msg)

    def _check_case_directories(self, definition_file_name, output_file_names):
        definition_file_path = self.case_dir / definition_file_name
        if not definition_file_path.is_file():
            raise FileNotFoundError(f"{definition_file_path} file not found within case {self.case_name}")
        self.case_output_dir.mkdir(exist_ok=True)
        if not self.overwrite and any([(self.case_output_dir/n).is_file() for n in output_file_names]):
            raise EnvironmentError(f"Case {case_name} already has results. To overwrite them use the --overwrite option")

    def call_symbolic_routine(self, routine_script_file_name, output_file_names):
        routine_script_file_path = self.case_dir / routine_script_file_name
        self._check_case_directories(routine_script_file_name, output_file_names)
        routine_module = importlib.import_module(f"{self.case_import_root}.{routine_script_file_path.stem}")
        compiler_outputs = routine_module.main(self.applied_config)
        for i in range(len(output_file_names)):
            with open(self.case_output_dir/output_file_names[i], 'w') as f:
                f.write(compiler_outputs[i])

    def run_fe_compiler(self):
        case_definition_file_name = "fe_definition.json"
        output_file_names = ["RHS.cpp", "LHS.cpp", "latex_prints.md", "cpp_interface.json"]
        self._check_case_directories(case_definition_file_name, output_file_names)
        compiler_outputs = KratosFECompiler(self.case_dir/case_definition_file_name).compile(self.applied_config)
        for i in range(len(output_file_names)):
            with open(self.case_output_dir/output_file_names[i], 'w') as f:
                f.write(compiler_outputs[i])
        
    def run_manual(self):
        routine_script_file_name = "manual_fe_generator.py"
        output_file_names = ["manual_RHS.cpp", "manual_LHS.cpp"]
        self.call_symbolic_routine(routine_script_file_name, output_file_names)
        
    def run_all(self):
        self.run_fe_compiler()
        self.run_manual()


if __name__ == "__main__":
    # parse options
    args = docopt(HELP, version=VERSION)

    type = args["--type"]
    source_dir = Path(args['--source'])
    case_name = args['CASE_NAME']
    overwrite = args['--overwrite']

    main_logger.set_logger(args['--quiet'], args['--debug'])

    symbolic_generator = SymbolicGenerator(source_dir, case_name, overwrite)

    generation_function = getattr(symbolic_generator,"run_"+type)
    generation_function()