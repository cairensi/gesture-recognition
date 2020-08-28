# Author by CRS-club and wizard

import os
import sys
import time
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
from datareader import get_reader

logging.root.handlers = []
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.DEBUG, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


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
        '--use_gpu',
        type=ast.literal_eval,
        default=False,
        help='default use gpu.')
    parser.add_argument(
        '--weights',
        type=str,
        default=None,
        help='weight path, None to use weights from Paddle.')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='sample number in a batch for inference.')
    parser.add_argument(
        '--filelist',
        type=str,
        default=None,
        help='path to inferenece data file lists file.')
    parser.add_argument(
        '--log_interval',
        type=int,
        default=1,
        help='mini-batch interval to log.')
    parser.add_argument(
        '--infer_topk',
        type=int,
        default=1,
        help='topk predictions to restore.')
    parser.add_argument(
        '--save_dir', type=str, default='./output', help='directory to store results')
    args = parser.parse_args()
    return args


def infer(args):
    # parse config
    config = parse_config(args.config)
    infer_config = merge_configs(config, 'infer', vars(args))
    print_configs(infer_config, "Infer")
    infer_model = models.get_model(args.model_name, infer_config, mode='infer')
    infer_model.build_input(use_pyreader=False)
    infer_model.build_model()
    infer_feeds = infer_model.feeds()
    infer_outputs = infer_model.outputs()
    label_dic = np.load('label.npy', allow_pickle=True).item()
    label_dic = {v:k for k,v in label_dic.items()}

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)

    filelist = args.filelist or infer_config.INFER.filelist
    assert os.path.exists(filelist), "{} not exist.".format(args.filelist)

    # get infer reader
    infer_reader = get_reader(args.model_name.upper(), 'infer', infer_config)

    if args.weights:
        assert os.path.exists(
            args.weights), "Given weight dir {} not exist.".format(args.weights)
    # if no weight files specified, exit()
    if args.weights:
        weights = args.weights
    else:
        print("model path must be specified")
        exit()
    infer_model.load_test_weights(exe, weights,
                                  fluid.default_main_program(), place)

    infer_feeder = fluid.DataFeeder(place=place, feed_list=infer_feeds)
    fetch_list = [x.name for x in infer_outputs]
    periods = []
    results = []
    cur_time = time.time()
    for infer_iter, data in enumerate(infer_reader()):
        data_feed_in = [items[:-1] for items in data]
        video_id = [items[-1] for items in data]
        infer_outs = exe.run(fetch_list=fetch_list,
                             feed=infer_feeder.feed(data_feed_in))
        prev_time = cur_time
        cur_time = time.time()
        period = cur_time - prev_time
        periods.append(period)
        # For classification model
        predictions = np.array(infer_outs[0])
        for i in range(len(predictions)):
            topk_inds = predictions[i].argsort()[0 - args.infer_topk:]
            topk_inds = topk_inds[::-1]
            preds = predictions[i][topk_inds]
            results.append(
                (video_id[i], preds.tolist(), topk_inds.tolist()))
        if args.log_interval > 0 and infer_iter % args.log_interval == 0:
            logger.info('Processed {} samples'.format((infer_iter + 1) * len(
                video_id)))

    logger.info('[INFER] infer finished. average time: {}'.format(
        np.mean(periods)))
    for result in results:
        result[2][0] = label_dic[result[2][0]]
        print([result[0], result[2], result[1]])
    # fluid.io.save_inference_model(dirname='infer_model', feeded_var_names=[infer_feeds[0].name],
    #                           target_vars=infer_outputs, executor=exe)


if __name__ == "__main__":
    args = parse_args()
    # check whether the installed paddle is compiled with GPU
    logger.info(args)

    infer(args)
