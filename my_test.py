import os.path as osp
import json
import os
import os.path as osp

import numpy as np
from scipy.sparse import csr_matrix
from scipy.io import loadmat
from sklearn.metrics import average_precision_score, precision_recall_curve

#import datasets
#from datasets.imdb import imdb
# from fast_rcnn.config import cfg
#from utils import cython_bbox
#from mydetector.datasets.imdb import imdb
from datasets.imdb import imdb

#utils/__init__.py
import _pickle as cPickle

def unpickle(file_path):
    with open(file_path, 'rb') as f:
        data = cPickle.load(f)
    return data


def pickle(data, file_path):
    with open(file_path, 'wb') as f:
        cPickle.dump(data, f, 0)


def gt_roidb(self):
    cache_file = osp.join(self.cache_path, self.name + '_gt_roidb.pkl')
    if osp.isfile(cache_file):
        roidb = unpickle(cache_file)
        return roidb

    # Load all images and build a dict from image to boxes
    all_imgs = loadmat(osp.join(self._root_dir, 'annotation', 'Images.mat'))
    all_imgs = all_imgs['Img'].squeeze()
    name_to_boxes = {}
    name_to_pids = {}
    for im_name, __, boxes in all_imgs:
        im_name = str(im_name[0])
        boxes = np.asarray([b[0] for b in boxes[0]])
        boxes = boxes.reshape(boxes.shape[0], 4)
        valid_index = np.where((boxes[:, 2] > 0) & (boxes[:, 3] > 0))[0]
        assert valid_index.size > 0, \
            'Warning: {} has no valid boxes.'.format(im_name)
        boxes = boxes[valid_index]
        name_to_boxes[im_name] = boxes.astype(np.int32)
        name_to_pids[im_name] = -1 * np.ones(boxes.shape[0], dtype=np.int32)

    def _set_box_pid(boxes, box, pids, pid):
        for i in range(boxes.shape[0]):
            if np.all(boxes[i] == box):
                pids[i] = pid
                return
        print('Warning: person {} box {} cannot find in Images'.format(pid, box))

    # Load all the train / test persons and label their pids from 0 to N-1
    # Assign pid = -1 for unlabeled background people
    if self._image_set == 'train':
        print('train=============================================')
        train = loadmat(osp.join(self._root_dir,
                                 'annotation/test/train_test/Train.mat'))
        train = train['Train'].squeeze()
        for index, item in enumerate(train):
            scenes = item[0, 0][2].squeeze()
            for im_name, box, __ in scenes:
                im_name = str(im_name[0])
                box = box.squeeze().astype(np.int32)
                _set_box_pid(name_to_boxes[im_name], box,
                             name_to_pids[im_name], index)
    else:
        print('test============================================')
        test = loadmat(osp.join(self._root_dir,
                                'annotation/test/train_test/TestG50.mat'))
        test = test['TestG50'].squeeze()
        for index, item in enumerate(test):
            # query
            im_name = str(item['Query'][0, 0][0][0])
            box = item['Query'][0, 0][1].squeeze().astype(np.int32)
            _set_box_pid(name_to_boxes[im_name], box,
                         name_to_pids[im_name], index)
            # gallery
            gallery = item['Gallery'].squeeze()
            for im_name, box, __ in gallery:
                im_name = str(im_name[0])
                if box.size == 0: break
                box = box.squeeze().astype(np.int32)
                _set_box_pid(name_to_boxes[im_name], box,
                             name_to_pids[im_name], index)

    # Construct the gt_roidb
    gt_roidb = []
    for im_name in self.image_index:
        boxes = name_to_boxes[im_name]
        boxes[:, 2] += boxes[:, 0]
        boxes[:, 3] += boxes[:, 1]
        pids = name_to_pids[im_name]
        num_objs = len(boxes)
        gt_classes = np.ones((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        overlaps[:, 1] = 1.0
        overlaps = csr_matrix(overlaps)
        gt_roidb.append({
            'boxes': boxes,
            'gt_classes': gt_classes,
            'gt_overlaps': overlaps,
            'gt_pids': pids,
            'flipped': False})
    pickle(gt_roidb, cache_file)
    print("wrote gt roidb to {}".format(cache_file))
    return gt_roidb

if __name__ == '__main__':
    #from datasets.psdb import psdb
    d = psdb('train',root_dir=r'F:\datasets\reid\CUHK-SYSU_V2\dataset')
    train = d.gt_roidb()
    print(len(train))
    d=psdb('test',root_dir=r'F:\datasets\reid\CUHK-SYSU_V2\dataset')
    test=d.gt_roidb()
    print(len(test))