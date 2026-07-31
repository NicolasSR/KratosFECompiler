import json

from lib.tensor_placeholders import NodalTensorPlaceholder, UnknownTensorPlaceholder, SymbolicTensorPlaceholder, ConstantTensorPlaceholder

def generate_interface_json(dim, nnodes_dict, placeholder_names_list, namespace, num_dofs):
    
    elem_space_names = list(nnodes_dict.keys())

    interface_dict = {
        "dim": dim,
        "nnodes_dict": nnodes_dict,
        "dofs": num_dofs,
        "vars": {
            "scalars": [],
            "nodal_scalars": {name: [] for name in elem_space_names},
            "vectors": [],
            "nodal_vectors": {name: [] for name in elem_space_names},
            "sym_matrices": [],
            "matrices": [],
            "rank_4_voigt": []
        }
    }

    vars_dict = interface_dict["vars"]
    for placeholder_name in placeholder_names_list:
        placeholder = namespace[placeholder_name]
        ph_type = type(placeholder)
        if ph_type == UnknownTensorPlaceholder or ph_type == NodalTensorPlaceholder:
            # First, get the element space name
            elem_space_name = placeholder.element_space_name
            nodes_name = placeholder.name+placeholder.nodes_name_complement
            if placeholder.rank == 0:
                vars_dict["nodal_scalars"][elem_space_name].append(nodes_name)
            elif placeholder.rank == 1:
                vars_dict["nodal_vectors"][elem_space_name].append(nodes_name)
        elif ph_type == SymbolicTensorPlaceholder or ph_type == ConstantTensorPlaceholder:
            gauss_name = placeholder.name+placeholder.gauss_name_complement
            if placeholder.rank == 0:
                vars_dict["scalars"].append(gauss_name)
            elif placeholder.rank == 1:
                vars_dict["vectors"].append(gauss_name)
            elif placeholder.rank == 2:
                if "symmetric" in placeholder.flags:
                    vars_dict["sym_matrices"].append(gauss_name)
                else:
                    vars_dict["matrices"].append(gauss_name)
            elif placeholder.rank == 4:
                if "use_voigt_notation" in placeholder.flags:
                    vars_dict["rank_4_voigt"].append(gauss_name)
                else:
                    raise NotImplemented("No implementation of non-voigt rank4 tensors yet")

    return json.dumps(interface_dict)

