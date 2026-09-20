from project_poet.load import load_inference


def get_model_and_config(path: str, device):
    return load_inference(path, device)
