from types import SimpleNamespace
import yaml


def model_config():

    try:
        with open("./configs/Unet.yaml", 'r', encoding="utf-8") as file:
            cfg = yaml.safe_load(file)
            model_cfg = SimpleNamespace(**cfg)
    except Exception as e:
        raise FileNotFoundError

    try:
        with open("./configs/Defualt.yaml", 'r', encoding="utf-8") as file:
            cfg = yaml.safe_load(file)
            para_cfg = SimpleNamespace(**cfg)
    except Exception as e:
        raise FileNotFoundError

    return model_cfg, para_cfg