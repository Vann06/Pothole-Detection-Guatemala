"""Reproducibility helpers and experiment id validation."""
import random
import re

import numpy as np
import torch

EXPERIMENT_ID = re.compile(r"[A-Za-z0-9_-]+")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_experiment_id(value):
    """Reject ids that could escape the reports/checkpoints folders."""
    # el id termina siendo nombre de carpeta, asi que nada de rutas raras
    if not isinstance(value, str) or not EXPERIMENT_ID.fullmatch(value):
        raise ValueError(f"experiment_id must match {EXPERIMENT_ID.pattern}: {value!r}")
    return value
