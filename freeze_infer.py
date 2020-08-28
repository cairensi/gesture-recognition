# Author by CRS-club and wizard

import paddle.fluid as fluid
from datareader import get_reader
import time
import numpy as np
from config import *
import argparse


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
    args = parser.parse_args()
    return args


args = parse_args()
config = parse_config(args.config)
label_dic = np.load('label.npy', allow_pickle=True).item()
label_dic = {v: k for k, v in label_dic.items()}
infer_config = merge_configs(config, 'infer', vars(args))
place = fluid.CPUPlace()
exe = fluid.Executor(place)
infer_program, feed_list, fetch_target = fluid.io.load_inference_model(dirname='infer_model',
                                                                       executor=exe)
feed_lists = [infer_program.global_block().var(feed_list[0])]
infer_reader = get_reader(args.model_name.upper(), 'infer', infer_config)
infer_feeder = fluid.DataFeeder(place=place, feed_list=feed_lists)
periods = []
results = []
cur_time = time.time()
for infer_iter, data in enumerate(infer_reader()):
    data_feed_in = [items[:-1] for items in data]
    video_id = [items[-1] for items in data]
    infer_outs = exe.run(infer_program,
                         fetch_list=fetch_target,
                         feed=infer_feeder.feed(data_feed_in))
    prev_time = cur_time
    cur_time = time.time()
    period = cur_time - prev_time
    periods.append(period)
    # For classification model
    predictions = np.array(infer_outs[0])
    for i in range(len(predictions)):
        topk_inds = predictions[i].argsort()[0 - 1:]
        topk_inds = topk_inds[::-1]
        preds = predictions[i][topk_inds]
        results.append(
            (video_id[i], preds.tolist(), topk_inds.tolist()))

logger.info('[INFER] infer finished. average time: {}'.format(
    np.mean(periods)))

for result in results:
    result[2][0] = label_dic[result[2][0]]
    print([result[0], result[2], result[1]])
