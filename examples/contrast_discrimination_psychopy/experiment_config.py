from datetime import datetime

import numpy as np


constants = {
    "savefolder": "./databases/",
    "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    "config_path": "./aepsych_config.ini",
    "seed": 1,
}

# base parameters in case we don't want AEPsych to manage all 8.
base_params = {
    "spatial_frequency": 2,
    "orientation": 0,
    "pedestal": 0.5,
    "contrast": 0.75,
    "temporal_frequency": 0,
    "size": 10,
    "angle_dist": 0,
    "eccentricity": 0,
}


psychopy_vars = {
    "setSizePix": [1680, 1050],
    "setWidth": 47.475,
    "setDistance": 57,
    "pre_duration_s": 0.0,
    "stim_duration_s": 5.0,
    "post_duration_s": 1,
    "response_wait": 2,
    "iti": 0,
}
