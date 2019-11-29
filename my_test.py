
import numpy as np
import torch
import numpy as np
#(2, 6978, 5)

import pickle

det_file='/a.rar'
all_boxes='[1],[2]1],[2]1],[2]1],[2]1],[2]'
with open(det_file, 'wb') as f:
    pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

