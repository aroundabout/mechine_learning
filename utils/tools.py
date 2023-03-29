import os
import yaml
import random
import numpy as np
import tensorflow as tf
from utils.const import CONFIG_FILENAME

def get_cur_path() -> str:
    return os.getcwd()

def read_config(yaml_name: str = CONFIG_FILENAME) -> dict:
    print(get_cur_path())
    config_file = open(os.path.join(get_cur_path(), yaml_name), 'r')
    return yaml.safe_load(config_file.read())
    
    
def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

if __name__=='__name__':
    print(get_cur_path())