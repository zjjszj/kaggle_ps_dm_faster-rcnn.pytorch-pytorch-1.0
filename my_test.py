
import numpy as np
import torch
import numpy as np
#(2, 6978, 5)

import pickle

a=np.array([[2.48685455e+02, 1.80785248e+02, 2.83983063e+02, 2.90654968e+02,
        9.93798077e-01],
       [4.25379578e+02, 1.72840088e+02, 4.78905090e+02, 3.50654816e+02,
        9.93560672e-01],
       [4.64349152e+02, 1.77561554e+02, 5.23979614e+02, 3.74912720e+02,
        9.89811957e-01],
       [0.00000000e+00, 0.00000000e+00, 1.35925570e+01, 2.63381081e+01,
        1.32355868e-04]])
det_thresh=0.5
for det in zip(a):
    det = np.asarray(det)
    print('det.shape={0}'.format(det.shape))  #det.shape=(72, 5)
    print('det={0}'.format(det))
    inds = np.where(det[:, 4].ravel() >= det_thresh)[0]
    det = det[inds]
    num_det = det.shape[0]
    print('num_det====',num_det)

