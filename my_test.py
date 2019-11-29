
import numpy as np
import torch
import numpy as np
#(2, 6978, 5)

import pickle

a=([[5.08841400e+02, 2.53884201e+01, 6.45735962e+02, 4.49250000e+02,
        9.84242439e-01],
       [1.52061356e+02, 7.82148972e+01, 2.85799255e+02, 4.36646851e+02,
        9.25699353e-01],
       [6.71549683e+01, 8.70890350e+01, 1.73165329e+02, 4.38227051e+02,
        8.24304640e-01],
       [4.05646454e+02, 1.13974205e+02, 4.90995667e+02, 3.69514496e+02,
        7.32826412e-01],
       [5.55549072e+02, 3.87783852e+01, 6.09612244e+02, 2.22234695e+02,
        6.72966003e-01],
       [4.57071960e+02, 3.08326691e+02, 4.73053040e+02, 3.53320618e+02,
        6.62475169e-01],
       [6.32870850e+02, 2.84004517e+02, 6.61417297e+02, 3.57643860e+02,
        3.35088521e-01],
       [1.24290710e+02, 7.82676697e+01, 1.83803955e+02, 2.73475800e+02,
        3.29978794e-01],
       [5.10878632e+02, 2.83012573e+02, 5.48353638e+02, 4.12611938e+02,
        1.58015504e-01],
       [3.80333252e+02, 1.49975876e+02, 4.30954926e+02, 3.30527283e+02,
        1.28268391e-01],
       [2.03751831e+02, 5.84589844e+01, 2.66718414e+02, 2.55020447e+02,
        1.26533195e-01],
       [1.60663681e+02, 4.47713013e+01, 2.23851501e+02, 2.60695251e+02,
        1.24875061e-01],
       [5.86513672e+02, 4.37518616e+01, 6.43140320e+02, 2.41771133e+02,
        1.19735561e-01],
       [4.46201569e+02, 1.35922089e+02, 4.88010406e+02, 2.78480652e+02,
        1.04566857e-01],
       [5.03941284e+02, 2.81198853e+02, 5.29899292e+02, 3.52555054e+02,
        9.35588181e-02],
       [4.85413605e+02, 8.60197296e+01, 5.71504944e+02, 3.93840179e+02,
        8.46241862e-02],
       [4.42732086e+02, 3.07427551e+02, 4.60209503e+02, 3.56483337e+02,
        8.42483118e-02],
       [6.22774902e+02, 7.25293045e+01, 6.39487061e+02, 1.19909363e+02,
        8.17935839e-02],
       [2.90259895e+01, 1.72109802e+02, 1.01784714e+02, 4.37153748e+02,
        7.61892423e-02],
       [6.15512085e+02, 6.61972580e+01, 6.48093628e+02, 1.77724518e+02,
        4.96760756e-02],
       [4.31849792e+02, 3.03833862e+02, 4.50295471e+02, 3.56154144e+02,
        4.43347394e-02],
       [3.21242218e+02, 5.45856400e+01, 4.06665833e+02, 3.52699341e+02,
        2.80664563e-02],
       [6.98795349e+02, 3.95883118e+02, 7.14544128e+02, 4.39775696e+02,
        2.64622252e-02],
       [6.29435913e+02, 2.53168488e+01, 7.37541748e+02, 3.49353546e+02,
        2.09742691e-02],
       [5.94428284e+02, 1.55190735e+02, 6.82290161e+02, 4.28624634e+02,
        1.42411757e-02],
       [2.27169769e+02, 1.48788879e+02, 3.46996002e+02, 4.45717163e+02,
        1.32952640e-02],
       [6.91155212e+02, 4.01775085e+02, 7.06337646e+02, 4.43883667e+02,
        1.29932119e-02],
       [2.39951630e+02, 9.70210190e+01, 2.71203308e+02, 1.95818329e+02,
        1.28869954e-02],
       [5.96795410e+02, 5.45111465e+01, 6.27051147e+02, 1.56260651e+02,
        1.27230063e-02],
       [7.89027832e+02, 3.73401855e+02, 7.99500000e+02, 4.42785004e+02,
        6.81018783e-03],
       [7.70474304e+02, 4.07373718e+02, 7.91598999e+02, 4.49250000e+02,
        6.39637699e-03],
       [2.96237122e+02, 0.00000000e+00, 3.76799622e+02, 1.66905365e+02,
        5.28805424e-03],
       [4.83541870e+02, 2.92674286e+02, 5.00442749e+02, 3.43033051e+02,
        3.92961875e-03],
       [5.59651245e+02, 4.19643259e+00, 5.94494507e+02, 4.15765152e+01,
        2.48336885e-03],
       [4.10078857e+02, 2.28123276e+02, 4.50561279e+02, 3.60955536e+02,
        2.30487855e-03],
       [6.72919434e+02, 3.68117554e+02, 6.91702881e+02, 4.26418945e+02,
        2.25120666e-03],
       [6.44691895e+02, 3.33006439e+02, 6.68837402e+02, 4.14727844e+02,
        1.79816619e-03],
       [4.39879089e+02, 2.70528564e+02, 4.61785645e+02, 3.33984680e+02,
        1.79088535e-03],
       [5.66260803e+02, 3.94450928e+02, 5.88490173e+02, 4.47153534e+02,
        9.99238342e-04],
       [7.88328857e+02, 0.00000000e+00, 7.99500000e+02, 2.87458191e+01,
        5.61500143e-04],
       [6.95824707e+02, 4.25854126e+02, 7.15475220e+02, 4.49250000e+02,
        4.85020282e-04],
       [6.55987427e+02, 4.15931458e+02, 6.78123047e+02, 4.49250000e+02,
        4.82538890e-04],
       [6.68803589e+02, 4.19749084e+02, 6.90218201e+02, 4.49250000e+02,
        4.80827905e-04],
       [5.81032837e+02, 3.91197449e+02, 6.03916443e+02, 4.49250000e+02,
        3.57364566e-04],
       [6.33683594e+02, 4.13556793e+02, 6.58702393e+02, 4.49250000e+02,
        2.73842248e-04],
       [5.94993469e+02, 3.90622620e+02, 6.19166870e+02, 4.49250000e+02,
        2.58666289e-04],
       [1.29884720e+00, 0.00000000e+00, 5.40793877e+01, 1.63125763e+02,
        2.05119912e-04],
       [3.37771988e+01, 0.00000000e+00, 5.76142349e+01, 2.14206276e+01,
        1.93208747e-04],
       [6.11833984e+02, 4.01659882e+02, 6.37058105e+02, 4.49250000e+02,
        1.90763123e-04],
       [5.82298203e+01, 0.00000000e+00, 8.20494766e+01, 2.05989819e+01,
        1.73353619e-04],
       [1.31864548e-02, 0.00000000e+00, 1.52175293e+01, 3.17825718e+01,
        1.68832630e-04],
       [0.00000000e+00, 4.11130493e+02, 8.81542683e+00, 4.49250000e+02,
        4.11310830e-05]])


def compare(a,det_thresh=0.5):
        det = np.asarray(a)
        print(det[:, 4].ravel() >= det_thresh)
        # inds = np.where()[0]
        # print('inds=',inds)
        # det = det[inds]
        # print('det=',det)
        # num_det = det.shape[0]
        # print('num_det====', num_det)
compare(a)