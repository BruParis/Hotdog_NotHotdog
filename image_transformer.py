import random
import numpy as np
from PIL import Image
from PIL import ImageFilter


class ImageTransformer():

    def _rotate(self, img):
        random_degree = random.uniform(-25, 25)
        return img.rotate(random_degree)

    def _noise(self, img):
        row, col = img.size
        ch = len(img.getbands())
        # normal(mean,sigma,...)
        gauss = np.random.normal(0, 30., (row, col, ch))
        noisy = np.clip(np.array(img) + gauss, 0, 255.)
        res = Image.fromarray(noisy.astype('uint8'), mode='RGB')
        return res

    def _h_flip(self, img):
        return img.transpose(Image.FLIP_LEFT_RIGHT)

    def _blur(self, img):
        return img.filter(ImageFilter.BLUR)

    def __init__(self):
        self.transformations = {
            'rotate': self._rotate,
            'noise': self._noise,
            'h_flip': self._h_flip,
            'blur': self._blur,
        }

    def transform_img(self, orig_img, trans_str=None):
        if trans_str is None:
            key = random.choice(list(self.transformations))
            trasnsformed_img = self.transformations[key](orig_img)
            return trasnsformed_img
        else:
            trans = self.transformations[trans_str]
            if trans is None:
                return None

            trasnsformed_img = trans(orig_img)
            return trasnsformed_img
