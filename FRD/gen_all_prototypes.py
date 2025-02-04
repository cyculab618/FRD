import random
import torch
import torch.multiprocessing
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
import numpy as np
from datasets import build_dataset

def get_all_prototypes(model, args, num_cls, region_thresh, vis_path, single_stage=False):
    if not os.path.exists(vis_path):
        os.makedirs(vis_path, exist_ok=True)
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

    pos_all_data = [[] for _ in range(num_cls)]
    neg_all_data = [[] for _ in range(num_cls)]

    with torch.no_grad():
        model = model.cuda()
        model.eval()
        # 1. collect feature embeddings
        for img_idx, pack in enumerate(data_loader_cluster, start=0):
            if img_idx % 1000 == 0:
                print('Generate prototypes in img of {}/{}'.format(img_idx, len(data_loader_cluster)))
            ms_imgs = []
            for scale_level in range(len(pack[0])//2):
                ms_imgs.append(torch.cat([pack[0][scale_level*2], pack[0][scale_level*2+1]], dim=0).cuda(non_blocking=True))
            tgt = pack[1][0].cuda()
            # ==================================================
            # 1-1. ss/ms inferring
            cams_list, feats_list = [], []
            for idx, img in enumerate(ms_imgs):
                # print(f"img.shape: {img.shape}")
                token_logit, patch_logit, feat, cam = model(ms_imgs[idx], forward_feat=True, use_fine=args.return_fine)
                cam = cam[:, :num_cls]
                if idx == 0:
                    size = cam.shape[-2:]
                else:
                    feat = F.interpolate(feat, size, mode='bilinear', align_corners=True)
                    cam = F.interpolate(cam, size, mode='bilinear', align_corners=True)

                feat = (feat[0] + feat[1].flip(-1))/2
                cam = (cam[0]+cam[1].flip(-1))/2
                cams_list.append(cam)
                feats_list.append(feat)
                if single_stage:
                    break
            ms_cam = torch.stack(cams_list).mean(0)  # [num_cls, H, W]
            ms_feat = torch.stack(feats_list).mean(0)  # [C, H, W]
            # ==================================================
            # 1-2. normalize
            norm_ms_cam = (ms_cam - ms_cam.flatten(-2).min(-1)[0][...,None,None]) / (ms_cam.flatten(-2).max(-1)[0][...,None,None]- ms_cam.flatten(-2).min(-1)[0][...,None,None] + 1e-3)
            # 1-3. collect regional features
            for cls_idx, is_exist in enumerate(tgt[:num_cls]):
                region_feat = (ms_feat[:, norm_ms_cam[cls_idx] > region_thresh])
                # print(f"region_feat.shape: {region_feat.shape[-1]}")
                if region_feat.shape[-1] < 1:
                    continue
                if is_exist:
                    pos_all_data[cls_idx].append(region_feat.mean(1))
                if not is_exist:
                    neg_all_data[cls_idx].append(region_feat.mean(1))
        
        torch.save({'pos': pos_all_data, 'neg': neg_all_data},
               '{}/all_prototypes.pth'.format(vis_path))
