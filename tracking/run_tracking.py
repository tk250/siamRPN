from __future__ import absolute_import
from got10k.experiments import *
from siamRPNBIG import TrackerSiamRPNBIG
from siamRPNBIG_rgbt import TrackerSiamRPNRGBT
from siamRPNBIG_late import TrackerSiamRPNLate
import argparse
import os
import json

parser = argparse.ArgumentParser(description='PyTorch SiameseRPN Tracking')

parser.add_argument('--tracker_path', default='/home/krautsct/RGB-t-Val', metavar='DIR',help='path to dataset')
parser.add_argument('--experiment_name', default='default', metavar='DIR',help='path to weight')
parser.add_argument('--net_path', default='../train/experiments/pretrained_dense/model/model_e200.pth', metavar='DIR',help='path to weight')
parser.add_argument('--visualize', default=True, help='visualize')

args = parser.parse_args()

if __name__ == '__main__':

    """Load the parameters from json file"""
    json_path = os.path.join('experiments/{}'.format(args.experiment_name), 'parameters.json')
    assert os.path.isfile(json_path), ("No json configuration file found at {}".format(json_path))
    with open(json_path) as data_file:
        params = json.load(data_file)

    '''setup tracker'''
    tracker1 = TrackerSiamRPNBIG(params, '../train/experiments/pretrained_RGB/model/model_e100.pth', name='SiamRPN_RGB_pretrained')
    tracker2 = TrackerSiamRPNRGBT(params, '../train/experiments/pretrained_dense/model/model_e200.pth', name='SiamRPN_Dense+pretraining_e200')
    tracker3 = TrackerSiamRPNRGBT(params, '../train/experiments/pretrained_dense/model/model_e100.pth', name='SiamRPN_Dense+pretraining_e100')
    tracker4 = TrackerSiamRPNRGBT(params, '../train/experiments/pretrained_dense_onlyRGBT234/model/model_e100.pth', name='RGBT234only_e100')
    tracker5 = TrackerSiamRPNRGBT(params, '../train/experiments/dense_fusion/model/model_e97.pth', name='SiamRPN_Dense_e100')
    #tracker6 = TrackerSiamRPNLate(params, '../train/experiments/late_fusion/model/model_e50.pth', name='SiamRPN_Late')


    '''setup experiments'''
    # 7 datasets with different versions
    '''
    experiments = ExperimentGOT10k('data/GOT-10k', subset='test'),
        ExperimentOTB('data/OTB', version=2015),
        ExperimentOTB('data/OTB', version=2013),
        ExperimentVOT('data/vot2018', version=2018),
        ExperimentUAV123('data/UAV123', version='UAV123'),
        ExperimentUAV123('data/UAV123', version='UAV20L'),
        ExperimentDTB70('data/DTB70'),
        ExperimentTColor128('data/Temple-color-128'),
        ExperimentNfS('data/nfs', fps=30),
        ExperimentNfS('data/nfs', fps=240),
    ]

    for e in experiments:
        e.run(tracker, visualize=True)
        e.report([tracker.name])
    '''

    '''
    experiments = ExperimentGOT10k('/home/krautsct/Downloads/full_data', subset='val',
                    result_dir='experiments/{}/GOT-10k_results'.format(args.experiment_name),
                    report_dir='experiments/{}/GOT-10k_reports'.format(args.experiment_name))

    '''
    experiments1 = ExperimentOTB('/home/krautsct/RGB-t-Val', version='test',
                    result_dir='experiments/{}/all_tracking'.format(args.experiment_name),
                    report_dir='experiments/{}/all_tracking_report'.format(args.experiment_name))
    experiments2 = ExperimentOTB('/home/krautsct/RGB-t-Val', RGBT=True, version='test',
                    result_dir='experiments/{}/all_tracking'.format(args.experiment_name),
                    report_dir='experiments/{}/all_tracking_report'.format(args.experiment_name))


    '''run experiments'''
    experiments1.run(tracker1, visualize=False)
    experiments2.run(tracker2, visualize=False)
    experiments2.run(tracker3, visualize=False)
    experiments2.run(tracker4, visualize=False)
    experiments2.run(tracker5, visualize=False)
    #experiments2.run(tracker6, visualize=False)
    experiments1.report([tracker1.name, tracker2.name, tracker3.name, tracker4.name, tracker5.name])
