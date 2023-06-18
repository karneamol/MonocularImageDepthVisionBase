import numpy as np
from utils import DepthNorm
from io import BytesIO
from PIL import Image
from zipfile import ZipFile
from tensorflow.keras.utils import Sequence
from augment import BasicPolicy
import csv


def extract_zip(input_zip):
    input_zip=ZipFile(input_zip)
    return {name: input_zip.read(name) for name in input_zip.namelist()}

def get_kitti_data(batch_size, kitti_data_zipfile='kitti_data.zip'):
    data = extract_zip(kitti_data_zipfile)

    # Here in the paths for csv file and paths of all image files (for rgb and depth pictures) within the csv file; it must be '/' and not '\' or "\\". Otherwise, will get KeyError.
    kitti_train = list((row.split(',') for row in (data['data/kitti_train.csv']).decode("utf-8").split('\n') if len(row) > 0))
    kitti_test = list((row.split(',') for row in (data['data/kitti_val.csv']).decode("utf-8").split('\n') if len(row) > 0))

    shape_rgb = (batch_size, 352, 1216, 3)
    shape_depth = (batch_size, 352, 1216, 1)

    # Helpful for testing...
    if False:
        kitti_train = kitti_train[:10]
        kitti_test = kitti_test[:10]

    return data, kitti_train, kitti_test, shape_rgb, shape_depth

def get_kitti_train_test_data(batch_size):
    data, kitti_train, kitti_test, shape_rgb, shape_depth = get_kitti_data(batch_size)

    train_generator = KITTI_BasicAugmentRGBSequence(data, kitti_train, batch_size=batch_size, shape_rgb=shape_rgb, shape_depth=shape_depth)
    test_generator = KITTI_BasicRGBSequence(data, kitti_test, batch_size=batch_size, shape_rgb=shape_rgb, shape_depth=shape_depth)

    return train_generator, test_generator

class KITTI_BasicAugmentRGBSequence(Sequence):
    def __init__(self, data, dataset, batch_size, shape_rgb, shape_depth, is_flip=False, is_addnoise=False, is_erase=False):
        self.data = data
        self.dataset = dataset
        self.policy = BasicPolicy( color_change_ratio=0.50, mirror_ratio=0.50, flip_ratio=0.0 if not is_flip else 0.2, 
                                    add_noise_peak=0 if not is_addnoise else 20, erase_ratio=-1.0 if not is_erase else 0.5)
        self.batch_size = batch_size
        self.shape_rgb = shape_rgb
        self.shape_depth = shape_depth
        self.maxDepth = 1000.0

        from sklearn.utils import shuffle
        self.dataset = shuffle(self.dataset, random_state=0)

        self.N = len(self.dataset)

    def __len__(self):
        return int(np.ceil(self.N / float(self.batch_size)))

    def __getitem__(self, idx, is_apply_policy=True):
        batch_x, batch_y = np.zeros( self.shape_rgb ), np.zeros( self.shape_depth )

        # Augmentation of RGB images
        for i in range(batch_x.shape[0]):
            index = min((idx * self.batch_size) + i, self.N-1)

            sample = self.dataset[index]

            x = np.clip(np.asarray(Image.open( BytesIO(self.data[sample[0]]) )).reshape(352,1216,3)/255,0,1)
            y = np.asarray(Image.open( BytesIO(self.data[sample[1].rstrip()])), dtype='float64').reshape(352,1216,1)
            
            y = np.where(y > 0, y, np.amax(y))
            y *= (1000.0/np.amax(y))
            y = DepthNorm(y, maxDepth=self.maxDepth)

            batch_x[i] = x
            batch_y[i] = y

            if is_apply_policy: batch_x[i], batch_y[i] = self.policy(batch_x[i], batch_y[i])

        return batch_x, batch_y

class KITTI_BasicRGBSequence(Sequence):
    def __init__(self, data, dataset, batch_size,shape_rgb, shape_depth):
        self.data = data
        self.dataset = dataset
        self.batch_size = batch_size
        self.N = len(self.dataset)
        self.shape_rgb = shape_rgb
        self.shape_depth = shape_depth
        self.maxDepth = 1000.0

    def __len__(self):
        return int(np.ceil(self.N / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x, batch_y = np.zeros( self.shape_rgb ), np.zeros( self.shape_depth )
        for i in range(self.batch_size):            
            index = min((idx * self.batch_size) + i, self.N-1)

            sample = self.dataset[index]

            x = np.clip(np.asarray(Image.open( BytesIO(self.data[sample[0]]))).reshape(352,1216,3)/255,0,1)
            y = np.asarray(Image.open( BytesIO(self.data[sample[1].rstrip()])), dtype='float64').reshape(352,1216,1)
            
            y = np.where(y > 0, y, np.amax(y))
            y *= (1000.0/np.amax(y))
            y = DepthNorm(y, maxDepth=self.maxDepth)

            batch_x[i] = x
            batch_y[i] = y

        return batch_x, batch_y