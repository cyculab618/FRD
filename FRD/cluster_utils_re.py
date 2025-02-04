import random

import torch
import torch.multiprocessing
from torch.multiprocessing import Manager
from torch.utils.data import DataLoader
# import voc12.dataloader
import torch.nn.functional as F
import cv2
from sklearn.cluster import KMeans
import os
import numpy as np
from datasets import build_dataset

def estimate_tensor_size(tensor):
    # 获取元素总数
    num_elements = tensor.numel()
    
    # 获取每个元素的字节大小
    element_size_bytes = tensor.element_size()
    
    # 计算总字节大小
    tensor_size_bytes = num_elements * element_size_bytes
    
    return tensor_size_bytes

cls_names = ['0_aeroplane', '1_bicycle', '2_bird', '3_boat', '4_bottle', '5_bus', '6_car', '7_cat', '8_chair',
             '9_cow', '10_diningtable', '11_dog', '12_horse', '13_motorbike', '14_person', '15_pottedplant',
             '16_sheep', '17_sofa', '18_train', '19_tvmonitor']
super_class = ['PERSON', 'ANIMAL', 'VEHICLE', 'INDOOR']
num_sub_class = {super_class[0]: 1, super_class[1]: 6, super_class[2]: 7, super_class[3]: 6}
super_class_map = {0: super_class[2], 1: super_class[2], 2: super_class[1], 3: super_class[2], 4: super_class[3],
                   5: super_class[2], 6: super_class[2], 7: super_class[1], 8: super_class[3], 9: super_class[1],
                   10: super_class[3], 11: super_class[1], 12: super_class[1], 13: super_class[2], 14: super_class[0],
                   15: super_class[3], 16: super_class[1], 17: super_class[3], 18: super_class[2], 19: super_class[3]}
landslide_cls_names = ['0_landslide']
ISPRS_cls_name  = ['0_Impervious surfaces', '1_Building', '2_Low vegetation', '3_Tree', '4_Car']
cityscape_cls_names = [
    '0_road', '1_sidewalk', '2_building', '3_wall', '4_fence', '5_pole',
    '6_traffic_light', '7_traffic_sign', '8_vegetation', '9_terrain', 
    '10_sky', '11_person', '12_rider', '13_car', '14_truck', 
    '15_bus', '16_train', '17_motorcycle', '18_bicycle'
]
cityscape_super_class = ['FLAT', 'HUMAN', 'VEHICLE', 'CONSTRUCTION', 'OBJECT', 'NATURE', 'SKY']
cityscape_num_sub_class = {cityscape_super_class[0]: 2, cityscape_super_class[1]: 2, cityscape_super_class[2]: 6, cityscape_super_class[3]: 3,
                           cityscape_super_class[4]: 3, cityscape_super_class[5]: 2, cityscape_super_class[6]: 1}
cityscape_super_class_map = {0: cityscape_super_class[0], 1: cityscape_super_class[0], 2: cityscape_super_class[3], 3: cityscape_super_class[3], 4: cityscape_super_class[3], 
                             5: cityscape_super_class[4], 6: cityscape_super_class[4], 7: cityscape_super_class[4], 8: cityscape_super_class[5], 9: cityscape_super_class[5], 
                             10: cityscape_super_class[6], 11: cityscape_super_class[1], 12: cityscape_super_class[1], 13: cityscape_super_class[2], 14: cityscape_super_class[2], 
                             15: cityscape_super_class[2], 16: cityscape_super_class[2], 17: cityscape_super_class[2], 18: cityscape_super_class[2]}
coco_cls_names = [
    '0_person', '1_bicycle', '2_car', '3_motorcycle', '4_airplane', '5_bus', '6_train', '7_truck', 
    '8_boat', '9_traffic_light', '10_fire_hydrant', '11_stop_sign', '12_parking_meter', '13_bench', 
    '14_bird', '15_cat', '16_dog', '17_horse', '18_sheep', '19_cow', '20_elephant', '21_bear', 
    '22_zebra', '23_giraffe', '24_backpack', '25_umbrella', '26_handbag', '27_tie', '28_suitcase', 
    '29_frisbee', '30_skis', '31_snowboard', '32_sports_ball', '33_kite', '34_baseball_bat', 
    '35_baseball_glove', '36_skateboard', '37_surfboard', '38_tennis_racket', '39_bottle', 
    '40_wine_glass', '41_cup', '42_fork', '43_knife', '44_spoon', '45_bowl', '46_banana', '47_apple', 
    '48_sandwich', '49_orange', '50_broccoli', '51_carrot', '52_hot_dog', '53_pizza', '54_donut', 
    '55_cake', '56_chair', '57_couch', '58_potted_plant', '59_bed', '60_dining_table', '61_toilet', 
    '62_tv', '63_laptop', '64_mouse', '65_remote', '66_keyboard', '67_cell_phone', '68_microwave', 
    '69_oven', '70_toaster', '71_sink', '72_refrigerator', '73_book', '74_clock', '75_vase', 
    '76_scissors', '77_teddy_bear', '78_hair_drier', '79_toothbrush'
]
coco_super_class = ['PERSON', 'VEHICLE', 'OUTDOOR', 'ANIMAL', 'ACCESSORY', 'SPORTS', 'KITCHEN', 'FOOD', 'FURNITURE', 'ELECTRONIC', 'APPLIANCE', 'INDOOR']
coco_num_sub_class = {
    coco_super_class[0]: 1, coco_super_class[1]: 8, coco_super_class[2]: 5, coco_super_class[3]: 10, coco_super_class[4]: 6,
    coco_super_class[5]: 10, coco_super_class[6]: 7, coco_super_class[7]: 12, coco_super_class[8]: 6, coco_super_class[9]: 6, 
    coco_super_class[10]: 6, coco_super_class[11]: 8
}
coco_super_class_map = {
    0: coco_super_class[0], 1: coco_super_class[1], 2: coco_super_class[1], 3: coco_super_class[1], 4: coco_super_class[1],
    5: coco_super_class[1], 6: coco_super_class[1], 7: coco_super_class[1], 8: coco_super_class[1], 9: coco_super_class[2],
    10: coco_super_class[2], 11: coco_super_class[2], 12: coco_super_class[2], 13: coco_super_class[2], 14: coco_super_class[3],
    15: coco_super_class[3], 16: coco_super_class[3], 17: coco_super_class[3], 18: coco_super_class[3], 19: coco_super_class[3],
    20: coco_super_class[3], 21: coco_super_class[3], 22: coco_super_class[3], 23: coco_super_class[3], 24: coco_super_class[4],
    25: coco_super_class[4], 26: coco_super_class[4], 27: coco_super_class[4], 28: coco_super_class[4], 29: coco_super_class[5],
    30: coco_super_class[5], 31: coco_super_class[5], 32: coco_super_class[5], 33: coco_super_class[5], 34: coco_super_class[5],
    35: coco_super_class[5], 36: coco_super_class[5], 37: coco_super_class[5], 38: coco_super_class[5], 39: coco_super_class[6],
    40: coco_super_class[6], 41: coco_super_class[6], 42: coco_super_class[6], 43: coco_super_class[6], 44: coco_super_class[6],
    45: coco_super_class[6], 46: coco_super_class[7], 47: coco_super_class[7], 48: coco_super_class[7], 49: coco_super_class[7],
    50: coco_super_class[7], 51: coco_super_class[7], 52: coco_super_class[7], 53: coco_super_class[7], 54: coco_super_class[7],
    55: coco_super_class[7], 56: coco_super_class[8], 57: coco_super_class[8], 58: coco_super_class[8], 59: coco_super_class[8],
    60: coco_super_class[8], 61: coco_super_class[8], 62: coco_super_class[9], 63: coco_super_class[9], 64: coco_super_class[9],
    65: coco_super_class[9], 66: coco_super_class[9], 67: coco_super_class[9], 68: coco_super_class[10], 69: coco_super_class[10],
    70: coco_super_class[10], 71: coco_super_class[10], 72: coco_super_class[10], 73: coco_super_class[11], 74: coco_super_class[11],
    75: coco_super_class[11], 76: coco_super_class[11], 77: coco_super_class[11], 78: coco_super_class[11], 79: coco_super_class[11]
}

def without_shared_feats(cls_idx, tgt, num_class):
    num_cls = len(tgt)
    if num_class == 19:
        return cityscape_super_class_map[cls_idx] not in [cityscape_super_class_map[i] for i in range(num_cls) if tgt[i]] and cityscape_num_sub_class[cityscape_super_class_map[cls_idx]]>1
    # TODO
    return super_class_map[cls_idx] not in [super_class_map[i] for i in range(num_cls) if tgt[i]] and num_sub_class[super_class_map[cls_idx]]>1


def list2tensor(feature_list):
    if len(feature_list) > 0:
        return torch.stack(feature_list)
    else:
        return torch.Tensor([])

# TODO SS/MS
def _get_cluster(model, args, num_cls, region_thresh, vis_path, num_cluster, single_stage=False, mask_img=False, sampling_ratio=1.):
    '''
    trainmsf_dataset = voc12.dataloader.VOC12ClassificationDatasetMSF("voc12/train_aug_id.txt",
                                                                      voc12_root="data/voc12/VOCdevkit/VOC2012",
                                                                      scales=(1.0, 0.5))
    
    trainmsf_dataloader = DataLoader(trainmsf_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True,
                                     drop_last=True)
    '''
    original_scale_parm = args.scales
    args.scales = (1.0, 0.5, 1.5)
    dataset_cluster, _ = build_dataset(is_train=False,data_set=args.data_set if 'cluster' in args.data_set else args.data_set + 'cluster', 
                                                     gen_attn=True, args=args)
    sampler_cluster = torch.utils.data.SequentialSampler(dataset_cluster)
    data_loader_cluster = torch.utils.data.DataLoader(dataset_cluster, sampler=sampler_cluster,
                                                      batch_size=1, num_workers=args.num_workers,
                                                      pin_memory=True, drop_last=True, shuffle=False)
    args.scales = original_scale_parm
    print(f"num_cls: {num_cls}")

    pos_clusters = [[] for _ in range(num_cls)]
    neg_info = [[] for _ in range(num_cls)]
    neg_clusters = [[] for _ in range(num_cls)]
    neg_clusters_unshared = [[] for _ in range(num_cls)]

    pos_all_data = [[] for _ in range(num_cls)]
    neg_all_data = [[] for _ in range(num_cls)]

    with torch.no_grad():
        model = model.cuda()
        model.eval()
        data_root = os.path.join(args.data_path, 'voc12', 'VOCdevkit', 'VOC2012')
        # 1. collect feature embeddings
        for img_idx, pack in enumerate(data_loader_cluster, start=0):
            if img_idx % 1000 == 0:
                print('processing regions features in img of {}/{}'.format(img_idx, len(data_loader_cluster)))
            ms_imgs = []
            for scale_level in range(len(pack[0])//2):
                ms_imgs.append(torch.cat([pack[0][scale_level*2], pack[0][scale_level*2+1]], dim=0).cuda(non_blocking=True))
            tgt = pack[1][0].cuda()
            # ==================================================
            # 1-1. ss/ms inferring
            logits_list, cams_list, feats_list = [], [], []
            for idx, img in enumerate(ms_imgs):
                # print(f"img.shape: {img.shape}")
                token_logit, patch_logit, feat, cam = model(ms_imgs[idx], forward_feat=True, use_fine=args.return_fine)
                cam = cam[:, :num_cls]
                if idx == 0:
                    size = cam.shape[-2:]
                else:
                    feat = F.interpolate(feat, size, mode='bilinear', align_corners=True)
                    cam = F.interpolate(cam, size, mode='bilinear', align_corners=True)

                logits_list.append(patch_logit.mean(0))
                feat = (feat[0] + feat[1].flip(-1))/2
                cam = (cam[0]+cam[1].flip(-1))/2
                cams_list.append(cam)
                feats_list.append(feat)
                if single_stage:
                    break
            ms_logits = torch.stack(logits_list).mean(0)
            ms_cam = torch.stack(cams_list).mean(0)  # [num_cls, H, W]
            ms_feat = torch.stack(feats_list).mean(0)  # [C, H, W]
            _, h, w = ms_cam.shape
            # ==================================================

            # 1-2. normalize
            # norm_ms_cam = F.relu(ms_cam) / (F.adaptive_max_pool2d(F.relu(ms_cam), (1, 1)) + 1e-5)
            norm_ms_cam = (ms_cam - ms_cam.flatten(-2).min(-1)[0][...,None,None]) / (ms_cam.flatten(-2).max(-1)[0][...,None,None]- ms_cam.flatten(-2).min(-1)[0][...,None,None] + 1e-3)
            # 1-3. collect regional features
            # orig_img = cv2.imread(os.path.join(data_root, 'JPEGImages', '{}.jpg'.format(name)))
            # orig_img = cv2.imread('data/voc12/VOCdevkit/VOC2012/JPEGImages/{}.jpg'.format(name))
            for cls_idx, is_exist in enumerate(tgt[:num_cls]):
                if is_exist:
                    region_feat = (ms_feat[:, norm_ms_cam[cls_idx] > region_thresh])
                    if region_feat.shape[-1] > 0:
                        pos_clusters[cls_idx].append(region_feat.mean(1))
                        pos_all_data[cls_idx].append(region_feat.mean(1))
                if not is_exist:
                    cam_mask = norm_ms_cam[cls_idx] > region_thresh
                    if cam_mask.sum() > 0:
                        info = [ms_logits[cls_idx], img_idx, cam_mask, tgt[:num_cls]]
                        neg_info[cls_idx].append(info)
                        region_feat = (ms_feat[:, norm_ms_cam[cls_idx] > region_thresh])
                        if region_feat.shape[-1] > 0:
                            neg_all_data[cls_idx].append(region_feat.mean(1))

        with open(f"{vis_path}/cls_feature_data.txt", "w") as f:
            pos_inner_pos_lengths = [len(inner_list) for inner_list in pos_clusters]
            f.write(f"pos_clusters's len: {pos_inner_pos_lengths}\n")
            neg_inner_pos_lengths = [len(inner_list) for inner_list in neg_info]
            f.write(f"neg_clusters's len: {neg_inner_pos_lengths}\n")
        torch.save({'pos': pos_all_data, 'neg': neg_all_data},
               '{}/all_prototypes.pth'.format(vis_path))
        # 2. get clusters from collected features
        for cls_idx in range(num_cls):
            
            # 2-1. positive cluster
            if len(pos_clusters[cls_idx]) > 0:
                pos_feats_np = torch.stack(pos_clusters[cls_idx]).cpu().numpy()
                num_k = min(num_cluster, len(pos_feats_np))
                centers = KMeans(n_clusters=num_k, random_state=0, max_iter=10).fit(pos_feats_np).cluster_centers_
                pos_clusters[cls_idx] = torch.from_numpy(centers).cuda()
            else:
                pos_clusters[cls_idx] = torch.Tensor([]).cuda()

            # 2-2. negative cluster
            if len(neg_info[cls_idx]) > 0:
                probs = torch.stack([item[0] for item in neg_info[cls_idx]])
                num_k = min(num_cluster, len(neg_info[cls_idx]))
                top_prob_idx = torch.topk(probs, num_k)[1]
                for top_i, item_idx in enumerate(top_prob_idx):
                    prob, img_idx, cam_mask, tgt = neg_info[cls_idx][item_idx]
                    prob = F.sigmoid(prob)
                    pack = data_loader_cluster.dataset.__getitem__(img_idx)
                    image_name = dataset_cluster.__getitem__(img_idx, return_name=True)[-1]
                    ss_img = pack[0][0]
                    img_mask = F.interpolate(cam_mask[None, None].float(), ss_img.shape[-2:], mode='nearest').cpu()[0]
                    masked_img = ss_img * img_mask
                    if num_cls == 1:
                        cv2.imwrite('{}/{}_{}_orig.png'.format(vis_path, landslide_cls_names[cls_idx], image_name),
                                    cv2.cvtColor(ss_img.numpy().transpose(1,2,0),cv2.COLOR_BGR2RGB) * 40 + 120)
                        cv2.imwrite('{}/{}_{}_prob{:.2f}.png'.format(vis_path, landslide_cls_names[cls_idx], image_name, prob),
                                    cv2.cvtColor(masked_img.numpy().transpose(1, 2, 0), cv2.COLOR_BGR2RGB) * 40 + 120)
                        _ ,_, feat, _ = model(ss_img[None].cuda(non_blocking=True), forward_feat=True, use_fine=args.return_fine)
                        feat = feat[0]
                        feat = (feat[:, cam_mask]).mean(1)
                        neg_clusters[cls_idx].append(feat)
                        neg_clusters_unshared[cls_idx].append(feat)
                    elif num_cls == 5:
                        cv2.imwrite('{}/{}_{}_orig.png'.format(vis_path, ISPRS_cls_name[cls_idx], image_name),
                                    cv2.cvtColor(ss_img.numpy().transpose(1,2,0),cv2.COLOR_BGR2RGB) * 40 + 120)
                        cv2.imwrite('{}/{}_{}_prob{:.2f}.png'.format(vis_path, ISPRS_cls_name[cls_idx], image_name, prob),
                                    cv2.cvtColor(masked_img.numpy().transpose(1, 2, 0), cv2.COLOR_BGR2RGB) * 40 + 120)
                        _ ,_, feat, _ = model(ss_img[None].cuda(non_blocking=True), forward_feat=True, use_fine=args.return_fine)
                        feat = feat[0]
                        feat = (feat[:, cam_mask]).mean(1)
                        neg_clusters[cls_idx].append(feat)
                        neg_clusters_unshared[cls_idx].append(feat)
                    elif num_cls == 19:
                        cv2.imwrite('{}/{}_{}_orig.png'.format(vis_path, cityscape_cls_names[cls_idx], image_name),
                                    cv2.cvtColor(ss_img.numpy().transpose(1,2,0),cv2.COLOR_BGR2RGB) * 40 + 120)
                        cv2.imwrite('{}/{}_{}_prob{:.2f}.png'.format(vis_path, cityscape_cls_names[cls_idx], image_name, prob),
                                    cv2.cvtColor(masked_img.numpy().transpose(1, 2, 0), cv2.COLOR_BGR2RGB) * 40 + 120)
                        _ ,_, feat, _ = model(ss_img[None].cuda(non_blocking=True), forward_feat=True, use_fine=args.return_fine)
                        feat = feat[0]
                        feat = (feat[:, cam_mask]).mean(1)
                        neg_clusters[cls_idx].append(feat)
                        if without_shared_feats(cls_idx, tgt):
                            neg_clusters_unshared[cls_idx].append(feat)
                    else:
                        cv2.imwrite('{}/{}_{}_orig.png'.format(vis_path, cls_names[cls_idx], image_name),
                                    cv2.cvtColor(ss_img.numpy().transpose(1,2,0),cv2.COLOR_BGR2RGB) * 40 + 120)
                        cv2.imwrite('{}/{}_{}_prob{:.2f}.png'.format(vis_path, cls_names[cls_idx], image_name, prob),
                                    cv2.cvtColor(masked_img.numpy().transpose(1, 2, 0), cv2.COLOR_BGR2RGB) * 40 + 120)

                        _ ,_, feat, _ = model(ss_img[None].cuda(non_blocking=True), forward_feat=True, use_fine=args.return_fine)
                        feat = feat[0]
                        feat = (feat[:, cam_mask]).mean(1)
                        neg_clusters[cls_idx].append(feat)
                        if without_shared_feats(cls_idx, tgt):
                            neg_clusters_unshared[cls_idx].append(feat)

                neg_clusters[cls_idx] = list2tensor(neg_clusters[cls_idx]).cuda()
                neg_clusters_unshared[cls_idx] = list2tensor(neg_clusters_unshared[cls_idx]).cuda()
            else:
                neg_clusters[cls_idx] = torch.Tensor([]).cuda()
                neg_clusters_unshared[cls_idx] = torch.Tensor([]).cuda()

    return pos_clusters, neg_clusters, neg_clusters_unshared


def get_regional_cluster(vis_path, model, args, num_cls=20, num_cluster=10, region_thresh=0.1,sampling_ratio=1.):
    """
    Args:
        model: the training model
        num_cls: the number of classes
        num_pos_k: the number of positive cluster for each class
        num_neg_k: the number of negative cluster for each class
        region_thresh: the threshold for getting reliable regions
    Return:
        clustered_fp_feats: a tensor with shape [num_cls, num_k, C]
    """

    if not os.path.exists(vis_path):
        os.makedirs(vis_path)

    pos_clusters, neg_clusters, neg_clusters_unshared = _get_cluster(model, args, num_cls, region_thresh, vis_path,
                                                                     num_cluster, sampling_ratio=sampling_ratio)

    torch.save({'pos': pos_clusters, 'neg': neg_clusters, 'neg_unshared': neg_clusters_unshared},
               '{}/clusters.pth'.format(vis_path))

    return pos_clusters, neg_clusters, neg_clusters_unshared
