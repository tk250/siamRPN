from __future__ import absolute_import
from got10k.experiments import *
from siamRPNBIG import TrackerSiamRPNBIG
import argparse
import os
import json

parser = argparse.ArgumentParser(description='PyTorch SiameseRPN Tracking')

parser.add_argument('--tracker_path', default='/home/krautsct/RGB-t-Val', metavar='DIR',help='path to dataset')
parser.add_argument('--experiment_name', default='default', metavar='DIR',help='path to weight')
parser.add_argument('--net_path', default='../train/experiments/default/model/model_e100.pth', metavar='DIR',help='path to weight')
parser.add_argument('--visualize', default=True, help='visualize')

args = parser.parse_args()

if __name__ == '__main__':

    """Load the parameters from json file"""
    json_path = os.path.join('experiments/{}'.format(args.experiment_name), 'parameters.json')
    assert os.path.isfile(json_path), ("No json configuration file found at {}".format(json_path))
    with open(json_path) as data_file:
        params = json.load(data_file)

    '''setup tracker'''
    tracker = TrackerSiamRPNBIG(params, args.net_path)

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
    experiments = ExperimentGOT10k(args.tracker_path, subset='val',
                    result_dir='experiments/{}/results'.format(args.experiment_name),
                    report_dir='experiments/{}/reports'.format(args.experiment_name))

    '''
    experiments = ExperimentOTB('/home/krautsct/RGB-t-Val', version='test',
                    result_dir='experiments/{}/OTB2015resultsRGBT'.format(args.experiment_name),
                    report_dir='experiments/{}/OTB2015reportsRGBT'.format(args.experiment_name))


    '''run experiments'''
    experiments.run(tracker, visualize=True)
    experiments.report([tracker.name])
