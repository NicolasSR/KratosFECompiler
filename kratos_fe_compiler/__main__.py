import importlib
from pathlib import Path

from docopt import docopt

from kratos_fe_compiler.metadata import VERSION
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
        self.case_import_root = f"{source_dir}.{case_name}"
        self.overwrite = overwrite

    def call_symbolic_routine(self, routine_script_file_name, output_file_names):
        routine_script_file_path = self.case_dir / routine_script_file_name
        if not routine_script_file_path.is_file():
            raise FileNotFoundError(f"{routine_script_file_name} file not found within case {self.case_name}")
        if not self.overwrite and any([(self.case_dir/n).is_file() for n in output_file_names]):
            raise EnvironmentError(f"Case {case_name} already has results. To overwrite them use the --overwrite option")
        routine_module = importlib.import_module(f"{self.case_import_root}.{routine_script_file_path.stem}")
        compiler_outputs = routine_module.main()
        for i in range(len(output_file_names)):
            with open(self.case_dir/output_file_names[i], 'w') as f:
                f.write(compiler_outputs[i])

    def run_fe_compiler(self):
        routine_script_file_name = "fe_definition.py"
        output_file_names = ["RHS.cpp", "LHS.cpp", "latex_prints.md", "cpp_interface.json"]
        self.call_symbolic_routine(routine_script_file_name, output_file_names)
        
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

    generation_funciton = getattr(symbolic_generator,"run_"+type)
    generation_funciton()