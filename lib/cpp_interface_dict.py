import json

from lib.tensor_placeholders import UnknownTensorPlaceholderRank0, UnknownTensorPlaceholderRank1, SymbolicTensorPlaceholderRank0
from lib.tensor_placeholders import SymbolicTensorPlaceholderRank1, SymbolicTensorPlaceholderRank2, SymbolicTensorPlaceholderRank4

def generate_interface_json(dim, nnodes, placeholders_list, num_dofs):
    interface_dict = {
        "dim": dim,
        "nnodes": nnodes,
        "dofs": num_dofs,
        "vars": {
            "scalars": [],
            "nodal_scalars": [],
            "vectors": [],
            "nodal_vectors": [],
            "sym_matrices": [],
            "matrices": [],
            "rank_4_voigt": []
        }
    }

    for placeholder in placeholders_list:
        vars_dict = interface_dict["vars"]
        ph_type = type(placeholder)
        if ph_type == UnknownTensorPlaceholderRank0:
            vars_dict["nodal_scalars"].append(placeholder.nodes_name)
        elif ph_type == SymbolicTensorPlaceholderRank0:
            vars_dict["scalars"].append(placeholder.gauss_name)
        elif ph_type == UnknownTensorPlaceholderRank1:
            vars_dict["nodal_vectors"].append(placeholder.nodes_name)
        elif ph_type == SymbolicTensorPlaceholderRank1:
            vars_dict["vectors"].append(placeholder.gauss_name)
        elif ph_type == SymbolicTensorPlaceholderRank2:
            if placeholder.flag_symmetric:
                vars_dict["sym_matrices"].append(placeholder.gauss_name)
            else:
                vars_dict["matrices"].append(placeholder.gauss_name)
        elif ph_type == SymbolicTensorPlaceholderRank4:
            if placeholder.flag_voigt_notation:
                vars_dict["rank_4_voigt"].append(placeholder.gauss_name)
            else:
                NotImplemented("No implementation of non-voigt rank4 tensors yet")
    
    # print(interface_dict)

    # free_vars = set()
    # for expr in expr_list:
    #     free_vars.update(expr.free_symbols)
    # print(free_vars)

    return json.dumps(interface_dict)