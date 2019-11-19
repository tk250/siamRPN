import numpy as np

class TrackerConfig(object):
    # These are the default hyper-params for DaSiamRPN 0.3827
    windowing = 'cosine'  # to penalize large displacements [cosine/uniform]
    # Params from the network architecture, have to be consistent with the training
    template_img_size = 127  # input z size
    detection_img_size = 271  # input x size (search region)
    total_stride = 8
    valid_scope = int((detection_img_size - template_img_size) / total_stride / 2)
    score_size = int((detection_img_size - template_img_size)/total_stride+1)
    context_amount = 0.5  # context amount for the exemplar
    ratios = [0.33, 0.5, 1, 2, 3]
    scales = [8, ]
    anchor_num = len(ratios) * len(scales)
    anchor = []
    penalty_k = 0.055
    window_influence = 0.42
    lr = 0.295
    lr_box = 0.30

    min_scale = 0.1
    max_scale = 10
    max_inter   = 80
    anchor_base_size = 8
    anchor_scales = np.array([8, ])
    anchor_ratios = np.array([0.33, 0.5, 1, 2, 3])
    size = anchor_num * score_size * score_size

    clip = 100 # grad clip

    anchor_scales = np.array([8, ])
    anchor_ratios = np.array([0.33, 0.5, 1, 2, 3])
    anchor_num    = len(anchor_scales) * len(anchor_ratios) # 5
    score_size = int((detection_img_size - template_img_size) / 8 + 1)
    size = anchor_num * score_size * score_size

    '''config for data.py'''

    out_feature = 19
    max_inter   = 80
    fix_former_3_layers = True
    pretrained_model =  None 

    anchor_total_stride = total_stride
    anchor_base_size = 8
    anchor_scales = np.array([8, ])
    anchor_ratios = np.array([0.33, 0.5, 1, 2, 3])

    valid_scope = int((detection_img_size - template_img_size) / total_stride / 2)
    anchor_valid_scope = 2 * valid_scope + 1
    pos_threshold = 0.6
    neg_threshold = 0.3

    context = 0.5
    penalty_k = 0.055
    window_influence = 0.42
    eps = 0.01

    max_translate = 12
    scale_resize = 0.15
    gray_ratio = 0.25
    exem_stretch = False

    '''config for net.py'''
    num_pos = 16
    num_neg = 48
    lamb    = 5

    ohem_pos = False
    ohem_neg = False
    ohem_reg = False

config = TrackerConfig()
