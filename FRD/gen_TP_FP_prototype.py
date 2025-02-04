import random
import torch
import torch.multiprocessing
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
import numpy as np
from datasets import build_dataset
from pathlib import Path
import imageio.v2 as imageio

def get_TP_FP_prototypes(model, args, num_cls, region_thresh, vis_path, single_stage=False):
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
    gt_dir = Path(args.data_path)/ "voc12" / "VOCdevkit" / "VOC2012" / "SegmentationClassAug"

    pos_all_data = [[] for _ in range(num_cls)]
    neg_all_data = [[] for _ in range(num_cls)]
    fn_all_data = [[] for _ in range(num_cls)]

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
            h_orig, w_orig = ms_imgs[0].shape[2], ms_imgs[0].shape[3]
            for idx, img in enumerate(ms_imgs):
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
            _, cam_h, cam_w = ms_cam.shape
            # ==================================================
            # 1-2. normalize
            norm_ms_cam = (ms_cam - ms_cam.flatten(-2).min(-1)[0][...,None,None]) / (ms_cam.flatten(-2).max(-1)[0][...,None,None]- ms_cam.flatten(-2).min(-1)[0][...,None,None] + 1e-3)
            cls_attentions = ms_cam.clone().unsqueeze(0)
            cls_attentions = F.interpolate(cls_attentions, size=(h_orig, w_orig), mode='bilinear', align_corners=False)
            cls_attentions = cls_attentions.squeeze(0)
            background_tensor = torch.full((1, h_orig, w_orig), region_thresh).cuda()
            total_attention = torch.cat((background_tensor, cls_attentions), dim=0)
            softmax_cls_attentions = torch.nn.functional.softmax(total_attention, dim=0)
            argmax_cls_attentions = torch.argmax(softmax_cls_attentions, dim=0)
            # ==================================================
            image_name = dataset_cluster.__getitem__(img_idx, return_name=True)[-1]
            gt_path = gt_dir / f"{image_name}.png"
            gt_img = imageio.imread(gt_path)
            gt_img = torch.tensor(gt_img).cuda()
            pred_result = argmax_cls_attentions.clone().cuda()
            # print(f"original image height: {h_orig}, width: {w_orig}")
            # print(f"cam height: {cam_h}, cam width: {cam_w}")
            # print(f"gt_img.shape: {gt_img.shape}, pred_result.shape: {pred_result.shape}")
            # 1-3. collect prototypes
            for cls_idx, is_exist in enumerate(tgt[:num_cls]):
                mask = torch.zeros(h_orig, w_orig).cuda()
                cls_id = cls_idx + 1
                mask[(gt_img == cls_id) & (pred_result == cls_id) & (gt_img != 255)] = 1
                mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0).float(), size=(cam_h, cam_w), mode='nearest').squeeze(0).squeeze(0)
                mask = mask.bool()
                gt_region_feat = (ms_feat[:, ((norm_ms_cam[cls_idx] > region_thresh) & (mask))])
                if gt_region_feat.shape[1] > 0:
                    pos_all_data[cls_idx].append(gt_region_feat.mean(1))
                # ==================================================
                tp_sum = (gt_img == cls_id) & (pred_result == cls_id) & (gt_img != 255)
                fp_sum = (gt_img != cls_id) & (pred_result == cls_id) & (gt_img != 255)
                tp_sum = tp_sum.sum()
                fp_sum = fp_sum.sum()
                if (tp_sum + fp_sum) > 0:
                    FP_rate = fp_sum / (fp_sum + tp_sum)
                else:
                    FP_rate = 0
                # print(f"tp_sum: {tp_sum}, fp_sum: {fp_sum}, FP_rate: {FP_rate}")
                # ==================================================
                fp_mask = torch.zeros(h_orig, w_orig).cuda()
                fp_mask[(gt_img != cls_id) & (pred_result == cls_id) & (gt_img != 255)] = 1
                fp_mask = F.interpolate(fp_mask.unsqueeze(0).unsqueeze(0).float(), size=(cam_h, cam_w), mode='nearest').squeeze(0).squeeze(0)
                fp_mask = fp_mask.bool()
                fp_region_feat = (ms_feat[:, ((norm_ms_cam[cls_idx] > region_thresh) & (fp_mask) & (mask==0))])
                if fp_region_feat.shape[1] > 0 and FP_rate > 0.5:
                    neg_all_data[cls_idx].append(fp_region_feat.mean(1))
                # ==================================================
                fn_mask = torch.zeros(h_orig, w_orig).cuda()
                fn_mask[(gt_img == cls_id) & (pred_result != cls_id) & (gt_img != 255)] = 1
                fn_mask = F.interpolate(fn_mask.unsqueeze(0).unsqueeze(0).float(), size=(cam_h, cam_w), mode='nearest').squeeze(0).squeeze(0)
                fn_mask = fn_mask.bool()
                fn_region_feat = (ms_feat[:, ((norm_ms_cam[cls_idx] > region_thresh) & (fn_mask) & (mask==0) & (fp_mask==0))])
                if fn_region_feat.shape[1] > 0:
                    fn_all_data[cls_idx].append(fn_region_feat.mean(1))
        
        torch.save({'pos': pos_all_data, 'neg': neg_all_data, 'fn': fn_all_data}, '{}/all_prototypes.pth'.format(vis_path))
        pos_inner_pos_lengths = [len(inner_list) for inner_list in pos_all_data]
        neg_inner_pos_lengths = [len(inner_list) for inner_list in neg_all_data]
        fn_inner_pos_lengths = [len(inner_list) for inner_list in fn_all_data]
        print(f"pos_inner_pos_lengths: {pos_inner_pos_lengths}")
        print(f"neg_inner_pos_lengths: {neg_inner_pos_lengths}")
        print(f"fn_inner_pos_lengths: {fn_inner_pos_lengths}")
        with open(f"{vis_path}/total_lengths.txt", "w") as f:
            f.write(f"pos_inner_pos_lengths: {pos_inner_pos_lengths}\n")
            f.write(f"neg_inner_pos_lengths: {neg_inner_pos_lengths}\n")
            f.write(f"fn_inner_pos_lengths: {fn_inner_pos_lengths}\n")
