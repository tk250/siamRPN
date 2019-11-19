from __future__ import absolute_import

import numpy as np
import time
import cv2
from PIL import Image

from ..utils.viz import show_frame


class Tracker(object):

    def __init__(self, name, is_deterministic=False):
        self.name = name
        self.is_deterministic = is_deterministic
    
    def init(self, image, box):
        raise NotImplementedError()

    def update(self, image):
        raise NotImplementedError()

    def track(self, img_files, infrared_files, box, visualize=False):
        frame_num = len(img_files)
        boxes = np.zeros((frame_num, 4))
        boxes[0] = box
        times = np.zeros(frame_num)

        for f, (img_file, infrared_file) in enumerate(zip(img_files, infrared_files)):
            image1 = Image.open(img_file)
            if not image1.mode == 'RGB':
                image1 = image1.convert('RGB')
            #image = np.array(image1)
            infrared_image = Image.open(infrared_file).convert('L')
            infrared_image = np.array(infrared_image) 

            start_time = time.time()
            if f == 0:
                self.init(image1, infrared_image, box)
            else:
                boxes[f, :] = self.update(image1, infrared_image)
            times[f] = time.time() - start_time

            if visualize:
                show_frame(image1, boxes[f, :])

        return boxes, times


from .identity_tracker import IdentityTracker
