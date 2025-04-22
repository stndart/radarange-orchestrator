import os

def find_model(path: str) -> str:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # Construct the path to your model file
    MODEL_PATH = os.path.join(BASE_DIR, '../models', path)
    return MODEL_PATH