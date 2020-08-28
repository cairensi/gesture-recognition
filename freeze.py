# Author by CRS-club and wizard

import os
import sys
import logging
import argparse
import ast
import numpy as np
try:
    import cPickle as pickle
except:
    import pickle
import paddle.fluid as fluid

from config import *
import models


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_name',
        type=str,
        default='tsn',
        help='name of model to train.')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/tsn.txt',
        help='path to config file of model')
    parser.add_argument(
        '--weights',
        type=str,
        default=None,
        help='weight path, None to use weights from Paddle.')
    args = parser.parse_args()
    return args


args = parse_args()
config = parse_config(args.config)
infer_config = merge_configs(config, 'infer', vars(args))
print_configs(infer_config, "Infer")
infer_model = models.get_model(args.model_name, infer_config, mode='infer')
infer_model.build_input(use_pyreader=False)
infer_model.build_model()
infer_feeds = infer_model.feeds()
infer_outputs = infer_model.outputs()
place = fluid.CPUPlace()
exe = fluid.Executor(place)

if args.weights:
    assert os.path.exists(
        args.weights), "Given weight dir {} not exist.".format(args.weights)
# if no weight files specified, exit
if args.weights:
    weights = args.weights
else:
    print("model path must be specified")
    exit()

infer_model.load_test_weights(exe, weights,
                              fluid.default_main_program(), place)
fluid.io.save_inference_model(dirname='infer_model', feeded_var_names=[infer_feeds[0].name],
                              target_vars=infer_outputs, executor=exe)
print('freeze done')
