from os import path

import pandas as pd
import numpy as np


def load_data(filename: str) -> pd.DataFrame:
    current_dir = path.dirname(path.realpath(__file__))
    data_dir = path.join(current_dir, '../../data/')

    return pd.read_csv(path.join(data_dir, f'{filename}.csv'))
    