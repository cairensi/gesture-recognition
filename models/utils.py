# Author by CRS-club and wizard

import os
import wget
import tarfile

__all__ = ['decompress', 'download', 'AttrDict']


def decompress(path):
    t = tarfile.open(path)
    t.extractall(path='/'.join(path.split('/')[:-1]))
    t.close()
    os.remove(path)


def download(url, path):
    weight_dir = '/'.join(path.split('/')[:-1])
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)

    path = path + ".tar.gz"
    wget.download(url, path)
    decompress(path)


class AttrDict(dict):
    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        if key in self.__dict__:
            self.__dict__[key] = value
        else:
            self[key] = value
