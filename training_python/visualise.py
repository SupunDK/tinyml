import numpy as np
import argparse

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from tqdm.auto import tqdm
from utils.help_code_demo import ToTensor, IEGM_DataSET, stats_report

parser = argparse.ArgumentParser()
# evaluation set-up params
parser.add_argument('--batch_size', type=int, default=1, help='batch size for inference')
parser.add_argument('--size', type=int, default=1250)
# data params
parser.add_argument('--path_data', type=str, default='../training_data/data_files/')
parser.add_argument('--path_indices', type=str, default='./data_indices')
parser.add_argument('--mode', type=str, default='total', choices=['total','train','eval','test'], help="evaluation mode.")
# classifier hyperparams
parser.add_argument('--subset', type=int, default=1250)
parser.add_argument('--factor', type=float, default=2.0)
parser.add_argument('--thresh', type=float, default=9)
parser.add_argument('--method', type=str, default='interval')
parser.add_argument('--ensemble', action="store_true", help="ensemble pred")
# logger params
parser.add_argument('--track', action="store_true", help="Whether to enable experiment trackers for logging.")
parser.add_argument('--tqdm_', action="store_true", help="tqdm viz")
parser.add_argument('--verbose', action="store_true", help="Verbose mode")
parser.add_argument('--print_freq', type=int, default = 0, help="print freq of perf data")
# instantiate
args = parser.parse_args()


def main():
    dataset = IEGM_DataSET(root_dir=args.path_data,
                            indice_dir=args.path_indices,
                            mode=args.mode,
                            size=args.size,
                            subject_id="S27",
                            transform=transforms.Compose([ToTensor()]))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    method = args.method
    subset = args.subset
    tqdm_ = args.tqdm_

    # ensemble method: optionally define the decision boundaries 
    if args.ensemble:
        factor =    [1.5, 1.75, 2.0, 2.25, 2.5]
        threshold = [9.191, 9.21, 9.263, 9.157, 9.154]
    else:
        factor = [args.factor]
        threshold = [args.thresh]
    visualise_peak_count_dist(dataloader, factor, method, threshold, subset, tqdm_)


def reject_outliers(data, m = 2.):
    '''
        reject outliers from the sampled peak intervals
        based on the variance away from the median value
    '''
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    ans = data[s<m]
    if ans.ndim != 1:
        ans = np.squeeze(ans)
    return ans


def countPeaks(chunk,factor,method,inp_len=1250):
    '''
        sample the number of detected peaks for a given input and the peak sampling threshold factor
    '''
    # only look at the subset of the input
    chunk = chunk['IEGM_seg'].squeeze()
    chunk = chunk[:inp_len]

    # find standard deviation and mean of the input
    std, mean = torch.std_mean(chunk, unbiased=False)
    if args.verbose:
        print(f"\nstd: {std} mean: {mean}")

    # detect flag prevents over-sampling near the peaks
    detect = True
    delay_steps = 0
    peak_cnt = 0
    peak_sampled_idx = []
    # iterate through the input and count peaks
    for idx, val in enumerate(chunk):
        peak_threshold = std.item() * factor
        if val > peak_threshold:
            if detect:
                peak_sampled_idx.append(idx)
                peak_cnt += 1
                delay_steps = 0
                detect = False
        if not detect:
            delay_steps += 1
            if delay_steps > 20:
                detect = True
    # truncation works only when three or more peaks are sampled
    if len(peak_sampled_idx) < 3:
        return len(peak_sampled_idx)
    # baseline method
    og_ans = len(peak_sampled_idx)
    # truncation method
    peak_diff = np.diff(np.asarray(peak_sampled_idx))
    trunc_peak_diff = reject_outliers(peak_diff)
    trunc_ans = trunc_peak_diff.size + 1
    # interval method
    robustAvgInterval = sum(trunc_peak_diff)/trunc_peak_diff.size
    inter_ans = inp_len/robustAvgInterval
    if args.verbose:
        print('sampled peak indices: ', peak_sampled_idx)
        print(f'baseline ans: {og_ans} | truncated ans: {trunc_ans} | interval ans: {inter_ans}')
    return inter_ans


def visualise_peak_count_dist(dataloader, factor, method, threshold_grid, subset=1250, tqdm_=False):
    AFb = []
    AFt = []
    SR = []
    SVT = []
    VFb = []
    VFt = []
    VPD = []
    VT = []

    for data in tqdm(dataloader):
        class_label = data['parsed_filename'][1][0]      
        peak_count = countPeaks(data,factor[0],method, subset)

        match class_label:
            case "AFb":
                AFb.append(peak_count)
            case "AFt":
                AFt.append(peak_count)
            case "SR":
                SR.append(peak_count)
            case "SVT":
                SVT.append(peak_count)
            case "VFb":
                VFb.append(peak_count)
            case "VFt":
                VFt.append(peak_count)
            case "VPD":
                VPD.append(peak_count)
            case "VT":
                VT.append(peak_count)
        
    overall = [AFb, AFt, SR, SVT, VFb, VFt, VPD, VT]

    x = []
    y = []

    for i, lis in enumerate(overall):
        x.extend([i]*len(lis))
        y.extend(lis)
    
    plt.plot(x, y)

if __name__ == '__main__':
    main()