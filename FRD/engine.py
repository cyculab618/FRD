import math
import sys
from typing import Iterable
import torch.nn as nn
import torch
import torch.nn.functional as F
import utils

from sklearn.metrics import average_precision_score
import numpy as np
import cv2
import os
from pathlib import Path

def train_one_epoch(model: torch.nn.Module, data_loader: Iterable,
                    args, optimizer: torch.optim.Optimizer, device: torch.device,
                    output_dir, epoch: int, loss_scaler, max_norm: float = 0,
                    set_training_mode=True):
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    num_class = args.nb_classes
    vis_path = os.path.join(output_dir, '{}_imgs_for_cluster'.format(epoch))
    
    model.train(set_training_mode)
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            # outputs = model(samples)
            cls_logit, patch_logit, feat, cams = model(samples, forward_feat=True, use_fine=args.return_fine)
            loss = F.multilabel_soft_margin_loss(cls_logit, targets)
            metric_logger.update(cls_loss=loss.item())
            ploss = F.multilabel_soft_margin_loss(patch_logit, targets)
            metric_logger.update(pat_loss=ploss.item())
            loss = loss + ploss

    
            loss_RC = torch.tensor(0.0).to(device)
            loss_PR = torch.tensor(0.0).to(device)
            metric_logger.update(loss_RC=loss_RC.item())
            metric_logger.update(loss_PR=loss_PR.item())
            
            if args.use_prototypes and epoch >= args.prototypes_start_epoch: #[FRD] prototypes loss
                if not args.prototypes_use_only_TP and not args.prototypes_use_only_FP: # use both
                    loss_choosen = 0
                elif args.prototypes_use_only_TP and not args.prototypes_use_only_FP: # use TP
                    loss_choosen = 1
                elif not args.prototypes_use_only_TP and args.prototypes_use_only_FP: # use FP
                    loss_choosen = 2
                loss_pos, loss_neg = get_prototypes_loss(device, patch_logit, feat, cams, targets, args.tempt,
                                                          args.mask_thresh, num_class,
                                                          args.prototypes_TP_use_patch, args.prototypes_FP_use_patch,
                                                          loss_choosen, args.prototypes_knum)
                prototypes_loss = loss_pos + loss_neg
                prototypes_loss = prototypes_loss * args.prototypes_weight
                metric_logger.update(loss_pos=loss_pos.item())
                metric_logger.update(loss_neg=loss_neg.item())
                metric_logger.update(prototypes_loss=prototypes_loss.item())
                loss = loss + prototypes_loss
            else:
                prototypes_loss = torch.tensor(0.0).to(device)
                metric_logger.update(prototypes_loss=prototypes_loss.item())

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.MultiLabelSoftMarginLoss()
    mAP = []

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        batch_size = images.shape[0]

        with torch.cuda.amp.autocast():
            output = model(images)
            if not isinstance(output, torch.Tensor):
                output, patch_output = output
            loss = criterion(output, target)
            output = torch.sigmoid(output)

            mAP_list = compute_mAP(target, output)
            # mAP = mAP + mAP_list
            metric_logger.meters['mAP'].update(np.mean(mAP_list), n=batch_size)

            ploss = criterion(patch_output, target)
            loss += ploss
            patch_output = torch.sigmoid(patch_output)

            mAP_list = compute_mAP(target, patch_output)
            # patch_mAP = patch_mAP + mAP_list
            metric_logger.meters['patch_mAP'].update(np.mean(mAP_list), n=batch_size)

        metric_logger.update(loss=loss.item())

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    print('* mAP {mAP.global_avg:.3f} loss {losses.global_avg:.3f}'
              .format(mAP=metric_logger.mAP, losses=metric_logger.loss))


    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def compute_mAP(labels, outputs):
    if labels.shape[-1] == 1:
        y_true = labels.cpu().numpy().ravel()
        y_pred = outputs.cpu().numpy().ravel()
        AP = average_precision_score(y_true, y_pred)
        return AP
    y_true = labels.cpu().numpy()
    y_pred = outputs.cpu().numpy()
    AP = []
    for i in range(y_true.shape[0]):
        if np.sum(y_true[i]) > 0:
            ap_i = average_precision_score(y_true[i], y_pred[i])
            AP.append(ap_i)
            # print(ap_i)
    return AP

def compute_cls(labels, outputs, threshold=0.5):
    y_true = labels.cpu().numpy()
    y_pred = outputs.cpu().numpy()
    num_classes = y_true.shape[1]
    # 初始化計數器
    TP = np.zeros(num_classes, dtype=np.int64)
    FP = np.zeros(num_classes, dtype=np.int64)
    TN = np.zeros(num_classes, dtype=np.int64)
    FN = np.zeros(num_classes, dtype=np.int64)
    for i in range(y_true.shape[0]):  
        for cls in range(num_classes):  
            if y_pred[i, cls] > threshold:  
                if y_true[i, cls] == 1:
                    TP[cls] += 1
                else:
                    FP[cls] += 1
            else: 
                if y_true[i, cls] == 1:
                    FN[cls] += 1
                else:
                    TN[cls] += 1
    
    return TP, FP, TN, FN

@torch.no_grad()
def generate_attention_maps_ms(data_loader, model, device, args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Generating attention maps:'
    if args.attention_dir is not None:
        Path(args.attention_dir).mkdir(parents=True, exist_ok=True)
    if args.cam_npy_dir is not None:
        Path(args.cam_npy_dir).mkdir(parents=True, exist_ok=True)

    # switch to evaluation mode
    model.eval()
    if 'coco' not in args.data_set.lower():
        img_list = open(os.path.join(args.data_path, args.img_list)).readlines()
    else:
        image_name = os.path.join(args.img_list, 'train_part_id.txt')
        img_list = open(image_name).readlines()
    # img_list = open(args.img_list).readlines()
    index = 0
    for image_list, target in metric_logger.log_every(data_loader, 10, header):
    # for iter, (image_list, target) in enumerate(data_loader):
        images1 = image_list[0].to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        batch_size = images1.shape[0]
        img_name = img_list[index].strip()
        index += 1

        img_temp = images1.permute(0, 2, 3, 1).detach().cpu().numpy()
        orig_images = np.zeros_like(img_temp)
        orig_images[:, :, :, 0] = (img_temp[:, :, :, 0] * 0.229 + 0.485) * 255.
        orig_images[:, :, :, 1] = (img_temp[:, :, :, 1] * 0.224 + 0.456) * 255.
        orig_images[:, :, :, 2] = (img_temp[:, :, :, 2] * 0.225 + 0.406) * 255.

        w_orig, h_orig = orig_images.shape[1], orig_images.shape[2]
        # w, h = images1.shape[2] - images1.shape[2] % args.patch_size, images1.shape[3] - images1.shape[3] % args.patch_size
        # w_featmap = w // args.patch_size
        # h_featmap = h // args.patch_size


        with torch.cuda.amp.autocast():
            cam_list = []
            vitattn_list = []
            cam_maps = None
            for s in range(len(image_list)):
                images = image_list[s].to(device, non_blocking=True)
                w, h = images.shape[2] - images.shape[2] % args.patch_size, images.shape[3] - images.shape[3] % args.patch_size
                w_featmap = w // args.patch_size
                h_featmap = h // args.patch_size

                if 'MCTformerV1' in args.model: 
                    output, cls_attentions, patch_attn = model(images, return_att=True, n_layers=args.layer_index)
                    cls_attentions = cls_attentions.reshape(batch_size, args.nb_classes, w_featmap, h_featmap)
                    patch_attn = torch.sum(patch_attn, dim=0)

                elif 'MCTformerV2' in args.model:            #cls_attentions = fusion  maps ，patch_attn = patch affinity maps
                    output, cls_attentions, patch_attn = model(images, return_att=True, n_layers=args.layer_index, attention_type=args.attention_type)
                    # print(patch_attn.shape)
                    patch_attn = torch.sum(patch_attn, dim=0)
                    # clone_patch_attn = patch_attn.clone().squeeze(0).cpu().numpy()
                    # clone_patch_attn = clone_patch_attn[-224:, -224:]
                    # vis_patch_attn(clone_patch_attn, os.path.join(args.cam_npy_dir, img_name + '_patch_attn.png')) # 產生patch attn圖片


                if args.patch_attn_refine:     #cls_attentions = refined fusion maps
                    cls_attentions = torch.matmul(patch_attn.unsqueeze(1), cls_attentions.view(cls_attentions.shape[0],cls_attentions.shape[1], -1, 1)).reshape(cls_attentions.shape[0],cls_attentions.shape[1], w_featmap, h_featmap)

                cls_attentions = F.interpolate(cls_attentions, size=(w_orig, h_orig), mode='bilinear', align_corners=False)[0]
                cls_attentions = cls_attentions.cpu().numpy() * target.clone().view(args.nb_classes, 1, 1).cpu().numpy()   # only visualize the positive classes
                # cls_attentions = cls_attentions.cpu().numpy()

                if s % 2 == 1:
                    cls_attentions = np.flip(cls_attentions, axis=-1)
                cam_list.append(cls_attentions)
                vitattn_list.append(cam_maps)

            sum_cam = np.sum(cam_list, axis=0)
            sum_cam = torch.from_numpy(sum_cam)
            sum_cam = sum_cam.unsqueeze(0).to(device)

            output = torch.sigmoid(output)

        if args.visualize_cls_attn:
            for b in range(images.shape[0]):
                if (target[b].sum()) >= 0:
                    cam_dict = {}
                    for cls_ind in range(args.nb_classes):
                        if target[b,cls_ind] >= 0:
                            cls_score = format(output[b, cls_ind].cpu().numpy(), '.3f')

                            cls_attention = sum_cam[b,cls_ind,:]

                            cls_attention = (cls_attention - cls_attention.min()) / (cls_attention.max() - cls_attention.min() + 1e-8)
                            cls_attention = cls_attention.cpu().numpy()

                            cam_dict[cls_ind] = cls_attention

                            if args.attention_dir is not None:
                                fname = os.path.join(args.attention_dir, img_name + '_' + str(cls_ind) + '_' + str(cls_score) + '.png')
                                show_cam_on_image(orig_images[b], cls_attention, fname)

                    if args.cam_npy_dir is not None:
                        np.save(os.path.join(args.cam_npy_dir, img_name + '.npy'), cam_dict)

                    if args.out_crf is not None:
                        for t in [args.low_alpha, args.high_alpha]:
                            orig_image = orig_images[b].astype(np.uint8).copy(order='C')
                            crf = _crf_with_alpha(cam_dict, t, orig_image)
                            folder = args.out_crf + ('_%s' % t)
                            if not os.path.exists(folder):
                                os.makedirs(folder)
                            np.save(os.path.join(folder, img_name + '.npy'), crf)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    return


def _crf_with_alpha(cam_dict, alpha, orig_img):
    from psa.tool.imutils import crf_inference
    v = np.array(list(cam_dict.values()))
    bg_score = np.power(1 - np.max(v, axis=0, keepdims=True), alpha)
    bgcam_score = np.concatenate((bg_score, v), axis=0)
    crf_score = crf_inference(orig_img, bgcam_score, labels=bgcam_score.shape[0])

    n_crf_al = dict()

    n_crf_al[0] = crf_score[0]
    for i, key in enumerate(cam_dict.keys()):
        n_crf_al[key + 1] = crf_score[i + 1]

    return n_crf_al


def show_cam_on_image(img, mask, save_path):
    img = np.float32(img) / 255.
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + img
    cam = cam / np.max(cam)
    cam = np.uint8(255 * cam)
    cv2.imwrite(save_path, cam)

def vis_patch_attn(patch_attn, output_dir):
    import matplotlib.pyplot as plt
    import seaborn as sns
    print(f"patch_attn shape: {patch_attn.shape}")
    plt.figure(figsize=(20, 20))
    ax = sns.heatmap(patch_attn, annot=False, cbar=False)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(output_dir)


def get_prototypes_loss(device, patch_logit, feats, cams, target, temperature, mask_threshold, num_class, TP_use_patch=False, FP_use_patch=False, loss_choosen=0, num_k=1):
    def exp_similarity(a, b, temperature):
        """
        calculate the distance of a, b. return the exp{-L1(a,b)}
        """
        if len(b) == 0 or len(a) == 0:  # no clusters
            return torch.Tensor([0.5]).to(a.device)

        # print(f"pos feat shape: {a.shape}, cls_norm_cam shape: {b.shape}")
        dis = ((a - b) ** 2 + 1e-4).mean(1)
        dis = torch.sqrt(dis)
        dis = dis / temperature + 0.1  # prevent too large gradient
        return torch.exp(-dis)
    def closest_dis(a, b):
        """
        Args:
            a: with shape of [1, C, HW]
            b: with shape of [num_clusters, C, 1]
        Return:
            dis: the distance of a, b. return the exp{-L1(a,b)}
        """
        if len(b) == 0 or len(a) == 0:  # no clusters
            return torch.Tensor([123456]).to(a.device)
        dis = ((a - b) ** 2 + 1e-4).mean(1)
        dis = dis.min(0)[0]
        dis = torch.sqrt(dis)
        dis = dis / temperature + 0.1  # prevent too large gradient
        return torch.exp(-dis)
    
    pos_loss = []
    neg_loss = []
    pos_prototypes = [[] for _ in range(num_class)]
    neg_prototypes = [[] for _ in range(num_class)]
    norm_cams = (cams - cams.flatten(-2).min(-1)[0][...,None,None]) / (cams.flatten(-2).max(-1)[0][...,None,None]- cams.flatten(-2).min(-1)[0][...,None,None] + 1e-3)
    norm_cams = (norm_cams > mask_threshold).detach().flatten(2)  # [B, K, HW]
    feats = feats.flatten(2)  # [B, C, HW]
    TP_per_iter = [0 for _ in range(num_class)]
    for feat, norm_cam, cam, gt in zip(feats, norm_cams, cams, target):
        for idx, is_exist in enumerate(gt):
            if cam[idx].max() <= 0 or norm_cam[idx].sum() == 0 or not is_exist:
                continue
            TP_per_iter[idx] += 1
            cls_norm_cam = norm_cam[idx]
            if TP_use_patch:
                region_feat = feat[:,cls_norm_cam]
                for i in range(region_feat.shape[1]):
                    if region_feat[:,i].sum() > 0:
                        pos_prototypes[idx].append(region_feat[:,i])
                        # print(f"pos_prototypes shape: {region_feat[:,i].shape}")
            else:
                region_feat = feat[:,cls_norm_cam].mean(-1)
                if region_feat.sum() > 0:
                    pos_prototypes[idx].append(region_feat)
    for cls_idx in range(num_class):
        probs_for_cls_idx = patch_logit[:, cls_idx]
        gt_for_cls_idx = target[:, cls_idx]
        neg_probs_for_cls_idx = probs_for_cls_idx[gt_for_cls_idx == 0]
        neg_cams = cams[gt_for_cls_idx == 0]
        neg_norm_cams = norm_cams[gt_for_cls_idx == 0]
        neg_out_features = feats[gt_for_cls_idx == 0]
        tp_num = TP_per_iter[cls_idx] * num_k
        num_k = min(len(neg_probs_for_cls_idx), tp_num)
        if num_k == 0:
            continue
        top_prob_idx = torch.topk(neg_probs_for_cls_idx, num_k)[1]
        for item_idx in top_prob_idx:
            if neg_cams[item_idx][cls_idx].max() <= 0 :
                continue
            cls_norm_cam = neg_norm_cams[item_idx][cls_idx]
            feat = neg_out_features[item_idx]
            if FP_use_patch:
                region_feat = feat[:,cls_norm_cam]
                for i in range(region_feat.shape[1]):
                    if region_feat[:,i].sum() > 0:
                        neg_prototypes[idx].append(region_feat[:,i])
            else:
                region_feat = feat[:,cls_norm_cam].mean(-1)
                if region_feat.sum() > 0:
                    neg_prototypes[cls_idx].append(region_feat)
    for cls_idx in range(num_class):
        if len(pos_prototypes[cls_idx]) == 0 or len(neg_prototypes[cls_idx]) == 0:
            continue
        current_cls_pos_prototypes = torch.stack(pos_prototypes[cls_idx])
        center_of_pos_prototypes = current_cls_pos_prototypes.mean(0, keepdim=True)
        if not True:
            current_cls_neg_prototypes = torch.stack(neg_prototypes[cls_idx])
            center_of_neg_prototypes = current_cls_neg_prototypes.mean(0, keepdim=True)
        for pos_prototype in pos_prototypes[cls_idx]:
            pos_prob = exp_similarity(pos_prototype[None], current_cls_pos_prototypes, temperature=temperature)
            pos_prob = pos_prob.mean()
            loss_pos = -torch.log(pos_prob)
            pos_loss.append(loss_pos)
        for neg_prototype in neg_prototypes[cls_idx]:
            neg_prob = exp_similarity(neg_prototype[None], center_of_pos_prototypes, temperature=temperature)
            neg_prob = neg_prob.mean()
            loss_neg = -torch.log(1 - neg_prob)
            neg_loss.append(loss_neg)
        # if FP_use_patch:
        #     for neg_prototype in neg_prototypes[cls_idx]:
        #         # print(f"neg_prototype shape: {neg_prototype[None].shape}, current_cls_pos_prototypes shape: {current_cls_pos_prototypes[..., None].shape}")
        #         neg_prob = closest_dis(neg_prototype[None], current_cls_pos_prototypes[..., None])
        #         neg_prob = neg_prob.mean()
        #         loss_neg = -torch.log(1 - neg_prob)
        #         neg_loss.append(loss_neg)
        # else:
        #     for neg_prototype in neg_prototypes[cls_idx]:
        #         neg_prob = exp_similarity(neg_prototype[None], center_of_pos_prototypes, temperature=temperature)
        #         loss_neg = -torch.log(1 - neg_prob)
        #         neg_loss.append(loss_neg)
    
    loss_pos = torch.stack(pos_loss).mean() if len(pos_loss) > 0 else torch.tensor(0.0).to(device)
    loss_neg = torch.stack(neg_loss).mean() if len(neg_loss) > 0 else torch.tensor(0.0).to(device)
    if loss_choosen == 0: # use both
        return loss_pos, loss_neg
    elif loss_choosen == 1: # use TP
        return loss_pos, torch.tensor(0.0).to(device)
    elif loss_choosen == 2: # use FP
        return torch.tensor(0.0).to(device), loss_neg
