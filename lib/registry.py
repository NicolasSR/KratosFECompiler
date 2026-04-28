from importlib.resources import files
import yaml

def load_operators():
    # Points to lib/data/operators.yaml
    path = files('lib.data').joinpath('operators.yaml')
    with path.open('r') as f:
        return yaml.safe_load(f)['operators']