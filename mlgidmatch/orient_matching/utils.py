from typing import List, Tuple, Any, Union
from dataclasses import dataclass, field
import numpy as np


@dataclass
class SimConfig:
    q_sim_3d: Union[np.ndarray, None] = None  # (peaks_num, 3)
    intens_sim_3d: Union[np.ndarray, None] = None  # (peaks_num, )
    rec: Union[np.ndarray, None] = None  # (3, 3)
