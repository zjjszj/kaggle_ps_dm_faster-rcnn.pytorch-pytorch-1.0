
import numpy as np
import torch
import numpy as np
#(2, 6978, 5)



det=[[[1,2,3,4],[1,2,3,4],[1,2,3,4]],[[1,2,3,4],[1,2,3,4],[1,2,3,4]]]
gt=[[[1,2,3,4],[1,2,3,4],[1,2,3,4]],[[1,2,3,4],[1,2,3,4],[1,2,3,4]]]

for g ,d in zip(det,gt):
    print('g===',g)
    print('d===',d)
# det=np.asarray(det)
# inds = np.where(det[:, 3].ravel() >= 0.5)[0]
# det = det[inds]
# num_det = det.shape[0]


# for i in det:
#     i = np.asarray(i)
#     print(i)
#     inds = np.where(i[:, 3].ravel() >= 0.5)[0]
#     i = i[inds]
#     num_det = i.shape[0]
#     print(num_det)