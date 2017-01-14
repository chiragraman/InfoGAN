__author__ = 'yuhongliang324'
import os
from PIL import Image
import numpy
import random

data_path = '../img_align_celeba'
celeba_npy = 'celeba_data.npy'


def collect(path=data_path, sample_rate=1., save_to=celeba_npy):
    files = os.listdir(path)
    files.sort()
    data = []
    count = 0
    for fn in files:
        if not fn.endswith('jpg'):
            continue
        r = random.random()
        if r > sample_rate:
            continue
        img_path = os.path.join(path, fn)
        img = Image.open(img_path)
        img = img.resize((28, 28), Image.ANTIALIAS)
        img = numpy.asarray(img)
        data.append(img)
        count += 1
        if count % 1000 == 0:
            print count
    data = numpy.stack(data, axis=0)
    print data.shape
    numpy.save(save_to, data)

if __name__ == '__main__':
    collect()

