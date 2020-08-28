# Author by CRS-club and wizard

import pickle
import cv2
import numpy as np
import random


class ReaderNotFoundError(Exception):
    "Error: reader not found"

    def __init__(self, reader_name, avail_readers):
        super(ReaderNotFoundError, self).__init__()
        self.reader_name = reader_name
        self.avail_readers = avail_readers

    def __str__(self):
        msg = "Reader {} Not Found.\nAvailiable readers:\n".format(
            self.reader_name)
        for reader in self.avail_readers:
            msg += "  {}\n".format(reader)
        return msg


class DataReader(object):
    """data reader for video input"""

    def __init__(self, model_name, mode, cfg):
        self.name = model_name
        self.mode = mode
        self.cfg = cfg

    def create_reader(self):
        """Not implemented"""
        pass

    def get_config_from_sec(self, sec, item, default=None):
        if sec.upper() not in self.cfg:
            return default
        return self.cfg[sec.upper()].get(item, default)


class ReaderZoo(object):
    def __init__(self):
        self.reader_zoo = {}

    def regist(self, name, reader):
        assert reader.__base__ == DataReader, "Unknow model type {}".format(
            type(reader))
        self.reader_zoo[name] = reader

    def get(self, name, mode, cfg):
        for k, v in self.reader_zoo.items():
            if k == name:
                return v(name, mode, cfg)
        raise ReaderNotFoundError(name, self.reader_zoo.keys())


# singleton reader_zoo
reader_zoo = ReaderZoo()


def regist_reader(name, reader):
    reader_zoo.regist(name, reader)


def get_reader(name, mode, cfg):
    reader_model = reader_zoo.get(name, mode, cfg)
    return reader_model.create_reader()
