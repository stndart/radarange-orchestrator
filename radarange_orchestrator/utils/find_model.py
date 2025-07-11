import os
from glob import glob


def find_model(path: str) -> list[str] | str:
    """
    Makes absolute path from a model filename
    """
    BASE_DIR = os.path.dirname(
        os.path.abspath('/root/radarange-orchestrator/utils/find_model.py')
    )
    # Construct the path to your model file
    model_candidate = os.path.join(BASE_DIR, '../models', path)
    models = glob(model_candidate)

    if len(models) == 1:
        return models[0]
    return models
