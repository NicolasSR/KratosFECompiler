import importlib
from pathlib import Path
import json
import re

from docopt import docopt

from kratos_fe_compiler.metadata import VERSION
from kratos_fe_compiler.compiler import KratosFECompiler
from lib.logger import main_logger
from kratos_fe_compiler.kratos_template_substitution_utils import PerformSubstitutionOnTemplate


PROGRAM_NAME = "kratos_fe_compiler"
HELP = """Sympy-based compiler for Finite Elements within KratosMultiphysics

Usage:
    {p} [options] CASE_NAME

Options:
    --overwrite         Overwrite output files if already exist

    --type=ARG          Type of generation to use ('fe_compiler', 'manual', 'all', 'kratos') [default: fe_compiler]

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
        
    def _extract_unique_ndims_nnodes_pairs(self, file_path):
        # Regex breakdown:
        # //substitute_{object_name}_ -> matches the literal text, {object_name} will be rhs or lhs
        # (\d+)              -> capture group 1: matches one or more digits for ndim
        # D                 -> matches literal "D_"
        # (\d+)              -> capture group 2: matches one or more digits for nnodes
        # N                  -> matches literal "N"

        pattern = {
            "rhs": r"//substitute_rhs_(\d+)D(\d+)N",
            "lhs": r"//substitute_lhs_(\d+)D(\d+)N"
        }
        unique_pairs = {
            "rhs": set(),
            "lhs": set()
        }

        def find_matches_for_object(line, object_name):
            matches = re.findall(pattern[object_name], line)
            for ndim, nnodes in matches:
                # Convert the string matches to integers and add as a tuple
                unique_pairs[object_name].add((int(ndim), int(nnodes)))

        try:
            with open(file_path, "r", encoding="utf-8") as file:
                for line in file:
                    # Find all matches in the current line
                    find_matches_for_object(line, "rhs")
                    find_matches_for_object(line, "lhs")
        except FileNotFoundError:
            raise Exception(f"Error: The file '{file_path}' was not found.")
        
        # Check that all the objects' pairs are the same, and that they are not empty
        objects_list = list(unique_pairs.keys())
        for object_name in objects_list[1:]:
            if unique_pairs[object_name]!=unique_pairs[objects_list[0]]:
                raise Exception(f"Error: The NDims and NNodes pairs for some objects among {str(objects_list)} differ")
        if len(unique_pairs[objects_list[0]])==0:
            raise Exception(f"Error: No substitution pattern found in template files")

        # Convert the set of tuples back to a list
        return list(unique_pairs[objects_list[0]])

    def call_symbolic_routine(self, routine_script_file_name, output_file_names):
        routine_script_file_path = self.case_dir / routine_script_file_name
        self._check_case_directories(routine_script_file_name, output_file_names)
        routine_module = importlib.import_module(f"{self.case_import_root}.{routine_script_file_path.stem}")
        compiler_outputs = routine_module.main(self.applied_config)
        for i in range(len(output_file_names)):
            with open(self.case_output_dir/output_file_names[i], 'w') as f:
                f.write(compiler_outputs[i])

    def run_fe_compiler(self):
        self.get_case_config()
        case_definition_file_name = "fe_definition.json"
        output_file_names = ["RHS.cpp", "LHS.cpp", "latex_prints.md", "cpp_interface.json"]
        self._check_case_directories(case_definition_file_name, output_file_names)
        compiler_outputs = KratosFECompiler(self.case_dir/case_definition_file_name).compile(self.applied_config)
        for i in range(len(output_file_names)):
            with open(self.case_output_dir/output_file_names[i], 'w') as f:
                f.write(compiler_outputs[i])
        
    def run_manual(self):
        self.get_case_config()
        routine_script_file_name = "manual_fe_generator.py"
        output_file_names = ["manual_RHS.cpp", "manual_LHS.cpp"]
        self.call_symbolic_routine(routine_script_file_name, output_file_names)

    def _process_element_or_condition(self, obj_name, template_path, output_path, case_definition_file_name):
        with open(template_path, "r") as template:
            output_text = template.read()
        ndims_nnodes_pairs = self._extract_unique_ndims_nnodes_pairs(template_path)
        for (ndims, nnodes) in ndims_nnodes_pairs:
            output_file_names = [
                f"RHS_{obj_name}_{ndims}D{nnodes}N.cpp",
                f"LHS_{obj_name}_{ndims}D{nnodes}N.cpp",
                f"latex_prints_{obj_name}_{ndims}D{nnodes}N.cpp.md",
                f"cpp_interface_{obj_name}_{ndims}D{nnodes}N.cpp.json"]
            self._check_case_directories(case_definition_file_name, output_file_names)
            applied_config = {"dim": ndims, "nnodes": nnodes}
            compiler_outputs = KratosFECompiler(self.case_dir/case_definition_file_name).compile(applied_config, functional_name = f"functional_{obj_name}")
            for i in range(len(output_file_names)):
                with open(self.case_output_dir/output_file_names[i], 'w') as f:
                    f.write(compiler_outputs[i])
            output_text = PerformSubstitutionOnTemplate(output_text, f"//substitute_rhs_{ndims}D{nnodes}N", self.case_output_dir/output_file_names[0])
            output_text = PerformSubstitutionOnTemplate(output_text, f"//substitute_lhs_{ndims}D{nnodes}N", self.case_output_dir/output_file_names[1])
        with open(output_path, "w") as element_output_file:
            element_output_file.write(output_text)

    def run_kratos(self):
        with open(self.case_dir/"case_config.json", 'r') as f:
            case_config = json.load(f)
        kratos_config = case_config["kratos_configuration"]
        case_definition_file_name = "fe_definition.json"

        # Process Element
        element_template_path = self.case_dir / kratos_config["element_template_path"]
        element_output_path = self.case_dir / kratos_config["element_output_path"]
        self._process_element_or_condition("element", element_template_path, element_output_path, case_definition_file_name)

        # Process Condition
        condition_template_path = self.case_dir / kratos_config["condition_template_path"]
        condition_output_path = self.case_dir / kratos_config["condition_output_path"]
        self._process_element_or_condition("condition", condition_template_path, condition_output_path, case_definition_file_name)
        
    def run_all(self):
        self.get_case_config()
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