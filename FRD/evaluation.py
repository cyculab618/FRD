import os
import pandas as pd
import numpy as np
from PIL import Image
import argparse

import torch
import torch.nn.functional as F

import joblib
from pathlib import Path

categories = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
              'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
              'tvmonitor']
categories_landslide = ['background', 'landslide']
categories_ISPRS  = ['background', 'Impervious surfaces', 'Building', 'Low vegetation', 'Tree', 'Car']
cityscape_cls_name  = ['background', 'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic_light', 'traffic_sign',
                       'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']
categories_coco = ['background',
                   'person',
                   'bicycle',
                   'car',
                   'motorcycle',
                   'airplane',
                   'bus',
                   'train',
                   'truck',
                   'boat',
                   'traffic light',
                   'fire hydrant',
                   'street sign',
                   'stop sign',
                   'parking meter',
                   'bench',
                   'bird',
                   'cat',
                   'dog',
                   'horse',
                   'sheep',
                   'cow',
                   'elephant',
                   'bear',
                   'zebra',
                   'giraffe',
                   'hat',
                   'backpack',
                   'umbrella',
                   'shoe',
                   'eye glasses',
                   'handbag',
                   'tie',
                   'suitcase',
                   'frisbee',
                   'skis',
                   'snowboard',
                   'sports ball',
                   'kite',
                   'baseball bat',
                   'baseball glove',
                   'skateboard',
                   'surfboard',
                   'tennis racket',
                   'bottle',
                   'plate',
                   'wine glass',
                   'cup',
                   'fork',
                   'knife',
                   'spoon',
                   'bowl',
                   'banana',
                   'apple',
                   'sandwich',
                   'orange',
                   'broccoli',
                   'carrot',
                   'hot dog',
                   'pizza',
                   'donut',
                   'cake',
                   'chair',
                   'couch',
                   'potted plant',
                   'bed',
                   'mirror',
                   'dining table',
                   'window',
                   'desk',
                   'toilet',
                   'door',
                   'tv',
                   'laptop',
                   'mouse',
                   'remote',
                   'keyboard',
                   'cell phone',
                   'microwave',
                   'oven',
                   'toaster',
                   'sink',
                   'refrigerator',
                   'blender',
                   'book',
                   'clock',
                   'vase',
                   'scissors',
                   'teddy bear',
                   'hair drier',
                   'toothbrush']


def do_python_eval(predict_folder, gt_folder, name_list, num_cls=21, input_type='png', threshold=1.0, printlog=False,
                   out_crf=False, n_jobs=10, img_dir=None, cam_type="cam", out_dir=None, return_fp_fn=False):
    def compare(idx):
        name = name_list[idx]
        gt_file = os.path.join(gt_folder, '%s.png' % name)
        gt = np.array(Image.open(gt_file))
        if input_type == 'png':
            predict_file = os.path.join(predict_folder, '%s.png' % name)
            predict = np.array(Image.open(predict_file))  # cv2.imread(predict_file)
            if num_cls == 81:
                predict = predict - 91
        elif input_type == 'npy':
            predict_file = os.path.join(predict_folder, '%s.npy' % name)
            predict_dict = np.load(predict_file, allow_pickle=True).item()

            cam = np.array([predict_dict[key] for key in predict_dict.keys()])
            label_key = np.array([key+1 for key in predict_dict.keys()]).astype(np.uint8)
            cam = np.pad(cam, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=threshold)
            label_key = np.pad(label_key, (1, 0), mode='constant', constant_values=0)

            cam = F.softmax(torch.tensor(cam).float(), dim=0).numpy()
            predict = np.argmax(cam, axis=0).astype(np.uint8)
            if predict.shape != gt.shape:
                predict = torch.tensor(predict).float()
                predict = F.interpolate(predict.unsqueeze(0).unsqueeze(0), size=(gt.shape[0], gt.shape[1]), mode='nearest').squeeze(0).squeeze(0).numpy()
                predict = np.array(predict).astype(np.uint8)
                # print(f"predict shape: {predict.shape}, gt shape: {gt.shape}")
            if out_crf:
                orig_image = np.array(Image.open(os.path.join(img_dir, name + '.jpg')).convert("RGB"))
                # crf use predict label
                if "cam" in cam_type:
                    from tool.imutils import crf_inference_label
                    predict = crf_inference_label(orig_image, predict, n_labels=cam.shape[0])
                elif "coco" in cam_type:
                    from tool.imutils import crf_inference_label_coco
                    predict = crf_inference_label_coco(orig_image, predict, n_labels=cam.shape[0])

            predict = label_key[predict]
            if out_dir is not None:
                predict_img = Image.fromarray(predict)
                predict_img.save(os.path.join(out_dir, name + ".png"))

        
        '''
        if gt.shape != (224, 224):
            print(gt_file)
            print(gt.shape)
        '''
        cal = gt < 255
        mask = (predict == gt) * cal

        p_list, t_list, tp_list = [0] * num_cls, [0] * num_cls, [0] * num_cls
        for i in range(num_cls):
            p_list[i] += np.sum((predict == i) * cal)
            t_list[i] += np.sum((gt == i) * cal)
            tp_list[i] += np.sum((gt == i) * mask)
        
        return p_list, t_list, tp_list

    results = joblib.Parallel(n_jobs=n_jobs, verbose=10, pre_dispatch="all")(
        [joblib.delayed(compare)(j) for j in range(len(name_list))]
    )
    p_lists, t_lists, tp_lists = zip(*results)
    TP = [0] * num_cls
    P = [0] * num_cls
    T = [0] * num_cls
    for idx in range(len(name_list)):
        p_list = p_lists[idx]
        t_list = t_lists[idx]
        tp_list = tp_lists[idx]
        for i in range(num_cls):
            TP[i] += tp_list[i]
            P[i] += p_list[i]
            T[i] += t_list[i]

    IoU = []
    T_TP = []
    P_TP = []
    FP_ALL = []
    FN_ALL = []
    Prediction = []
    Recall = []
    '''
    for i in range(num_cls):
        IoU.append(TP[i] / (T[i] + P[i] - TP[i]))
        T_TP.append(T[i] / (TP[i]))
        P_TP.append(P[i] / (TP[i]))
        FP_ALL.append((P[i] - TP[i]) / (T[i] + P[i] - TP[i]))
        FN_ALL.append((T[i] - TP[i]) / (T[i] + P[i] - TP[i]))
        Prediction.append(TP[i] / P[i])
        Recall.append(TP[i] / T[i])
    '''
    for i in range(num_cls):
      # Check if denominator is non-zero before performing division for IoU
      if (T[i] + P[i] - TP[i]) != 0:
          IoU.append(TP[i] / (T[i] + P[i] - TP[i]))
      else:
          IoU.append(0)  # or some other default value or logic

      # Check for T_TP
      if TP[i] != 0:
          T_TP.append(T[i] / TP[i])
      else:
          T_TP.append(0)  # or some other default value or logic

      # Check for P_TP
      if TP[i] != 0:
          P_TP.append(P[i] / TP[i])
      else:
          P_TP.append(0)  # or some other default value or logic

      # Check for FP_ALL
      if (T[i] + P[i] - TP[i]) != 0:
          FP_ALL.append((P[i] - TP[i]) / (T[i] + P[i] - TP[i]))
      else:
          FP_ALL.append(0)  # or some other default value or logic

      # Check for FN_ALL
      if (T[i] + P[i] - TP[i]) != 0:
          FN_ALL.append((T[i] - TP[i]) / (T[i] + P[i] - TP[i]))
      else:
          FN_ALL.append(0)  # or some other default value or logic

      # Check for Prediction
      if P[i] != 0:
          Prediction.append(TP[i] / P[i])
      else:
          Prediction.append(0)  # or some other default value or logic

      # Check for Recall
      if T[i] != 0:
          Recall.append(TP[i] / T[i])
      else:
          Recall.append(0)  # or some other default value or logic

    loglist = {}
    for i in range(num_cls):
        if num_cls == 21:
            loglist[categories[i]] = IoU[i] * 100
        elif num_cls == 20:
            loglist[cityscape_cls_name[i]] = IoU[i] * 100
        elif num_cls == 2:
            loglist[categories_landslide[i]] = IoU[i] * 100
        elif num_cls == 6:
            loglist[categories_ISPRS[i]] = IoU[i] * 100
        else:
            loglist[categories_coco[i]] = IoU[i] * 100
    miou = np.nanmean(np.array(IoU))
    loglist['mIoU'] = miou * 100
    fp = np.nanmean(np.array(FP_ALL))
    loglist['FP'] = fp * 100
    fn = np.nanmean(np.array(FN_ALL))
    loglist['FN'] = fn * 100
    prediction = np.nanmean(np.array(Prediction))
    loglist['Prediction'] = prediction * 100
    recall = np.nanmean(np.array(Recall))
    loglist['Recall'] = recall * 100
    if printlog:
        for i in range(num_cls):
            if num_cls == 21:
                if i % 2 != 1:
                    print('%11s:%7.3f%%' % (categories[i], IoU[i] * 100), end='\t')
                else:
                    print('%11s:%7.3f%%' % (categories[i], IoU[i] * 100))
            elif num_cls == 20:
                if i % 2 != 1:
                    print('%11s:%7.3f%%' % (cityscape_cls_name[i], IoU[i] * 100), end='\t')
                else:
                    print('%11s:%7.3f%%' % (cityscape_cls_name[i], IoU[i] * 100))
            elif num_cls == 2:
                if i % 2 != 1:
                    print('%11s:%7.3f%%' % (categories_landslide[i], IoU[i] * 100), end='\t')
                else:
                    print('%11s:%7.3f%%' % (categories_landslide[i], IoU[i] * 100))
            elif num_cls == 6:
                if i % 2 != 1:
                    print('%11s:%7.3f%%' % (categories_ISPRS[i], IoU[i] * 100), end='\t')
                else:
                    print('%11s:%7.3f%%' % (categories_ISPRS[i], IoU[i] * 100))
            else:
                if i % 2 != 1:
                    print('%11s:%7.3f%%' % (categories_coco[i], IoU[i] * 100), end='\t')
                else:
                    print('%11s:%7.3f%%' % (categories_coco[i], IoU[i] * 100))
        print('\n======================================================')
        print('%11s:%7.3f%%' % ('mIoU', miou * 100))
        print('\n')
        print(f'FP = {fp * 100}, FN = {fn * 100}')
        print(f'Prediction = {prediction * 100}, Recall = {recall * 100}')
    if not return_fp_fn:
        return loglist
    else:
        return loglist, FP_ALL, FN_ALL, IoU


def writedict(file, dictionary):
    s = ''
    for key in dictionary.keys():
        sub = '%s:%s  ' % (key, dictionary[key])
        s += sub
    s += '\n'
    file.write(s)


def writelog(filepath, metric, comment):
    filepath = filepath
    logfile = open(filepath, 'a')
    import time
    logfile.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    logfile.write('\t%s\n' % comment)
    writedict(logfile, metric)
    logfile.write('=====================================\n')
    logfile.close()


def whole_eval(list, predict_dir, gt_dir, logfile, comment='comment', type='npy',
               t=None, curve=True, num_classes=21, start=0, end=60):
    if type == 'npy':
        assert t is not None or curve
    df = pd.read_csv(list, names=['filename'], converters={"filename": str})
    name_list = df['filename'].values
    if not curve:
        loglist = do_python_eval(predict_dir, gt_dir, name_list, num_classes, type, t, printlog=True)
        writelog(logfile, loglist, comment)
    else:
        l = []
        max_mIoU = 0.0
        best_thr = 0.0
        for i in range(start, end):
            t = i / 100.0
            loglist = do_python_eval(predict_dir, gt_dir, name_list, num_classes, type, t, printlog=True)
            l.append(loglist['mIoU'])
            print('%d/%d background score: %.3f\tmIoU: %.3f%%' % (i, end, t, loglist['mIoU']))
            if loglist['mIoU'] > max_mIoU:
                max_mIoU = loglist['mIoU']
                best_thr = t
            else:
                break
        print('Best background score: %.3f\tmIoU: %.3f%%' % (best_thr, max_mIoU))
        writelog(logfile, {'mIoU': l, 'Best mIoU': max_mIoU, 'Best threshold': best_thr}, comment)

    return max_mIoU, best_thr


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--list", default='voc12/train_id.txt', type=str)
    parser.add_argument("--data-set", default='VOC12', type=str)
    parser.add_argument("--data-path", default='data/voc12')
    parser.add_argument("--predict_dir", default='./out_rw', type=str)
    parser.add_argument("--gt_dir", default='SegmentationClass', type=str)
    parser.add_argument("--img_dir", type=str, default=None)
    parser.add_argument("--out-crf", action="store_true")
    parser.add_argument("--out-dir", default=None, type=str)
    parser.add_argument('--logfile', default='./evallog.txt', type=str)
    # parser.add_argument('--comment', required=True, type=str)
    parser.add_argument('--type', default='png', choices=['npy', 'png'], type=str)
    parser.add_argument('--t', default=None, type=float)
    parser.add_argument('--curve', default=False, type=bool)
    parser.add_argument('--num_classes', default=21, type=int)
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--end', default=60, type=int)
    parser.add_argument('--n_jobs', default=10, type=int)
    parser.add_argument('--cam-type', default="cam", type=str)
    parser.add_argument('--record', action="store_true", default=True)
    parser.add_argument('--no-record', action="store_false", dest="record")
    parser.add_argument('--comment', default="eval_record", type=str)
    parser.add_argument('--ignore-bkg', action="store_true")

    args = parser.parse_args()

    if "coco" not in args.data_set:
        args.list = Path(args.data_path)  / args.list
        args.data_path = Path(args.data_path) / "voc12" if "voc12" not in args.data_path else args.data_path
        args.data_path = Path(args.data_path) / "VOCdevkit" / "VOC2012"
        args.gt_dir = args.data_path / "SegmentationClassAug"
        args.img_dir = args.data_path / "JPEGImages"
    if "coco" in args.data_set:
        args.data_path = Path(args.data_path) / "coco" if "coco" not in args.data_path else args.data_path
        args.list = Path(args.data_path)  / 'train_part_id.txt'
        args.gt_dir = args.data_path / "class_labels"
        args.img_dir = args.data_path / "train2014"
        args.cam_type="coco"
    
    if args.data_set == "VOC12":
        args.num_classes = 21
    elif args.data_set == "postdam":
        args.num_classes = 6
    elif args.data_set == "landslide":
        args.num_classes = 2
    elif args.data_set == "coco":
        args.num_classes = 81
    elif args.data_set == "cityscapes":
        args.num_classes = 20

    if args.type == 'npy':
        assert args.t is not None or args.curve
    df = pd.read_csv(args.list, names=['filename'], converters={"filename": str})
    name_list = df['filename'].values
    if args.t is not None:
        args.t = args.t / 100.0
    if args.out_dir is not None:
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    if not args.curve:
        loglist = do_python_eval(args.predict_dir, args.gt_dir, name_list, args.num_classes, args.type, args.t,
                                 printlog=True, cam_type=args.cam_type,
                                 out_crf=args.out_crf, n_jobs=args.n_jobs, img_dir=args.img_dir,
                                 out_dir=args.out_dir)
        # writelog(args.logfile, loglist, args.comment)
    else:
        l = []
        max_mIoU = 0.0
        best_thr = 0.0
        best_fp_rate = 0.0
        best_fn_rate = 0.0
        best_score = {}
        best_fp = []
        best_fn = []
        for i in range(args.start, args.end):
            t = i / 100.0
            loglist, FP_ALL, FN_ALL, IoU = do_python_eval(args.predict_dir, args.gt_dir, name_list, args.num_classes, args.type, t,
                                                          printlog=False, cam_type=args.cam_type,
                                                          out_crf=args.out_crf, n_jobs=args.n_jobs, img_dir=args.img_dir, return_fp_fn=True)
            l.append(loglist['mIoU'])
            print('%d/%d threshold: %.3f\tmIoU: %.3f%%' % (i, args.end, t, loglist['mIoU']))
            if loglist['mIoU'] > max_mIoU:
                max_mIoU = loglist['mIoU']
                best_thr = t
                best_score = loglist
                best_fp = FP_ALL
                best_fn = FN_ALL
                best_fp_rate = loglist['FP']
                best_fn_rate = loglist['FN']
            else:
                break
        fp_rate = 0.0
        fn_rate = 0.0
        miou = 0.0
        print('Best threshold: %.3f\tmIoU: %.3f%%' % (best_thr, max_mIoU))
        print('FP rate for each class: FP formula = (P - TP) / (T + P - TP)')
        for i in range(args.num_classes):
            if args.num_classes == 21:
                if i % 2 != 1:
                    print('%11s:%7.3f%%' % (categories[i], best_fp[i] * 100), end='\t')
                else:
                    print('%11s:%7.3f%%' % (categories[i], best_fp[i]  * 100))
            elif args.num_classes == 20:
                if i % 2 != 1:
                    print('%11s:%7.3f%%' % (cityscape_cls_name[i], best_fp[i] * 100), end='\t')
                else:
                    print('%11s:%7.3f%%' % (cityscape_cls_name[i], best_fp[i] * 100))
            elif args.num_classes == 2:
                if i % 2 != 1:
                    print('%11s:%7.3f%%' % (categories_landslide[i], best_fp[i] * 100), end='\t')
                else:
                    print('%11s:%7.3f%%' % (categories_landslide[i], best_fp[i] * 100))
            elif args.num_classes == 6:
                if i % 2 != 1:
                    print('%11s:%7.3f%%' % (categories_ISPRS[i], best_fp[i] * 100), end='\t')
                else:
                    print('%11s:%7.3f%%' % (categories_ISPRS[i], best_fp[i] * 100))
            else:
                if i % 2 != 1:
                    print('%11s:%7.3f%%' % (categories_coco[i], best_fp[i] * 100), end='\t')
                else:
                    print('%11s:%7.3f%%' % (categories_coco[i], best_fp[i] * 100))
            if i != 0:
                fp_rate += (best_fp[i] * 100)
        if args.ignore_bkg:
            best_fp_rate = fp_rate / (args.num_classes - 1)
        print('\n======================================================')
        print('%11s:%7.3f%%' % ('FP', best_fp_rate))
        print('')
        print('FN rate for each class: FN formula = (T - TP) / (T + P - TP)')
        for i in range(args.num_classes):
            if args.num_classes == 21:
                if i % 2 != 1:
                    print('%11s:%7.3f%%' % (categories[i], best_fn[i] * 100), end='\t')
                else:
                    print('%11s:%7.3f%%' % (categories[i], best_fn[i] * 100))
            elif args.num_classes == 20:
                if i % 2 != 1:
                    print('%11s:%7.3f%%' % (cityscape_cls_name[i], best_fn[i] * 100), end='\t')
                else:
                    print('%11s:%7.3f%%' % (cityscape_cls_name[i], best_fn[i] * 100))
            elif args.num_classes == 2:
                if i % 2 != 1:
                    print('%11s:%7.3f%%' % (categories_landslide[i], best_fn[i] * 100), end='\t')
                else:
                    print('%11s:%7.3f%%' % (categories_landslide[i], best_fn[i] * 100))
            elif args.num_classes == 6:
                if i % 2 != 1:
                    print('%11s:%7.3f%%' % (categories_ISPRS[i], best_fn[i] * 100), end='\t')
                else:
                    print('%11s:%7.3f%%' % (categories_ISPRS[i], best_fn[i] * 100))
            else:
                if i % 2 != 1:
                    print('%11s:%7.3f%%' % (categories_coco[i], best_fn[i] * 100), end='\t')
                else:
                    print('%11s:%7.3f%%' % (categories_coco[i], best_fn[i] * 100))
            if i != 0:
                fn_rate += (best_fn[i] * 100)
        if args.ignore_bkg:
            best_fn_rate = fn_rate / (args.num_classes - 1)
        print('\n======================================================')
        print('%11s:%7.3f%%' % ('FN', best_fn_rate))
        print('')
        print('mIoU for each class: IoU formula = TP / (T + P - TP)')
        for i in range(args.num_classes):
            if args.num_classes == 21:
                if i % 2 != 1:
                    print('%11s:%7.3f%%' % (categories[i], best_score[categories[i]]), end='\t')
                else:
                    print('%11s:%7.3f%%' % (categories[i], best_score[categories[i]]))
                if i != 0:
                    miou += best_score[categories[i]]
            elif args.num_classes == 20:
                if i % 2 != 1:
                    print('%11s:%7.3f%%' % (cityscape_cls_name[i], best_score[cityscape_cls_name[i]]), end='\t')
                else:
                    print('%11s:%7.3f%%' % (cityscape_cls_name[i], best_score[cityscape_cls_name[i]]))
                if i != 0:
                    miou += best_score[cityscape_cls_name[i]]
            elif args.num_classes == 2:
                if i % 2 != 1:
                    print('%11s:%7.3f%%' % (categories_landslide[i], best_score[categories_landslide[i]]), end='\t')
                else:
                    print('%11s:%7.3f%%' % (categories_landslide[i], best_score[categories_landslide[i]]))
                if i != 0:
                    miou += best_score[categories_landslide[i]]
            elif args.num_classes == 6:
                if i % 2 != 1:
                    print('%11s:%7.3f%%' % (categories_ISPRS[i], best_score[categories_ISPRS[i]]), end='\t')
                else:
                    print('%11s:%7.3f%%' % (categories_ISPRS[i], best_score[categories_ISPRS[i]]))
                if i != 0:
                    miou += best_score[categories_ISPRS[i]]
            else:
                if i % 2 != 1:
                    print('%11s:%7.3f%%' % (categories_coco[i], best_score[categories_coco[i]]), end='\t')
                else:
                    print('%11s:%7.3f%%' % (categories_coco[i], best_score[categories_coco[i]]))
                if i != 0:
                    miou += best_score[categories_coco[i]]
        if args.ignore_bkg:
            max_mIoU = miou / (args.num_classes - 1)
        print('\n======================================================')
        print('%11s:%7.3f%%' % ('mIoU', max_mIoU))
        print(f"Prediction = {best_score['Prediction']}, Recall = {best_score['Recall']}")
        # writelog(args.logfile, {'mIoU': l, 'Best mIoU': max_mIoU, 'Best threshold': best_thr}, args.comment)
        if args.record:
            path_parts = args.predict_dir.split('/')
            path_parts = path_parts[:-1]
            out_path = '/'.join(path_parts)
            record_file_name = os.path.join(out_path, args.comment + '.txt')
            with open (record_file_name, 'w') as f:
                f.write('Best threshold: %.3f\tmIoU: %.3f%%' % (best_thr, max_mIoU))
                f.write('\nFP rate for each class: FP formula = (P - TP) / (T + P - TP)\n')
                for i in range(args.num_classes):
                    if args.num_classes == 21:
                        if i % 2 != 1:
                            f.write('%11s:%7.3f%%\t' % (categories[i], best_fp[i] * 100))
                        else:
                            f.write('%11s:%7.3f%%\n' % (categories[i], best_fp[i] * 100))
                    elif args.num_classes == 20:
                        if i % 2 != 1:
                            f.write('%11s:%7.3f%%\t' % (cityscape_cls_name[i], best_fp[i] * 100))
                        else:
                            f.write('%11s:%7.3f%%\n' % (cityscape_cls_name[i], best_fp[i] * 100))
                    elif args.num_classes == 2:
                        if i % 2 != 1:
                            f.write('%11s:%7.3f%%\t' % (categories_landslide[i], best_fp[i] * 100))
                        else:
                            f.write('%11s:%7.3f%%\n' % (categories_landslide[i], best_fp[i] * 100))
                    elif args.num_classes == 6:
                        if i % 2 != 1:
                            f.write('%11s:%7.3f%%\t' % (categories_ISPRS[i], best_fp[i] * 100))
                        else:
                            f.write('%11s:%7.3f%%\n' % (categories_ISPRS[i], best_fp[i] * 100))
                    else:
                        if i % 2 != 1:
                            f.write('%11s:%7.3f%%\t' % (categories_coco[i], best_fp[i] * 100))
                        else:
                            f.write('%11s:%7.3f%%\n' % (categories_coco[i], best_fp[i] * 100))
                f.write('\n======================================================\n')
                f.write('%11s:%7.3f%%' % ('FP', best_fp_rate))
                f.write('\n\nFN rate for each class: FN formula = (T - TP) / (T + P - TP)\n')
                for i in range(args.num_classes):
                    if args.num_classes == 21:
                        if i % 2 != 1:
                            f.write('%11s:%7.3f%%\t' % (categories[i], best_fn[i] * 100))
                        else:
                            f.write('%11s:%7.3f%%\n' % (categories[i], best_fn[i] * 100))
                    elif args.num_classes == 20:
                        if i % 2 != 1:
                            f.write('%11s:%7.3f%%\t' % (cityscape_cls_name[i], best_fn[i] * 100))
                        else:
                            f.write('%11s:%7.3f%%\n' % (cityscape_cls_name[i], best_fn[i] * 100))
                    elif args.num_classes == 2:
                        if i % 2 != 1:
                            f.write('%11s:%7.3f%%\t' % (categories_landslide[i], best_fn[i] * 100))
                        else:
                            f.write('%11s:%7.3f%%\n' % (categories_landslide[i], best_fn[i] * 100))
                    elif args.num_classes == 6:
                        if i % 2 != 1:
                            f.write('%11s:%7.3f%%\t' % (categories_ISPRS[i], best_fn[i] * 100))
                        else:
                            f.write('%11s:%7.3f%%\n' % (categories_ISPRS[i], best_fn[i] * 100))
                    else:
                        if i % 2 != 1:
                            f.write('%11s:%7.3f%%\t' % (categories_coco[i], best_fn[i] * 100))
                        else:
                            f.write('%11s:%7.3f%%\n' % (categories_coco[i], best_fn[i] * 100))
                f.write('\n======================================================\n')
                f.write('%11s:%7.3f%%' % ('FN', best_fn_rate))
                f.write('\n\nmIoU for each class: IoU formula = TP / (T + P - TP)\n')
                for i in range(args.num_classes):
                    if args.num_classes == 21:
                        if i % 2 != 1:
                            f.write('%11s:%7.3f%%\t' % (categories[i], best_score[categories[i]]))
                        else:
                            f.write('%11s:%7.3f%%\n' % (categories[i], best_score[categories[i]]))
                    elif args.num_classes == 20:
                        if i % 2 != 1:
                            f.write('%11s:%7.3f%%\t' % (cityscape_cls_name[i], best_score[cityscape_cls_name[i]]))
                        else:
                            f.write('%11s:%7.3f%%\n' % (cityscape_cls_name[i], best_score[cityscape_cls_name[i]]))
                    elif args.num_classes == 2:
                        if i % 2 != 1:
                            f.write('%11s:%7.3f%%\t' % (categories_landslide[i], best_score[categories_landslide[i]]))
                        else:
                            f.write('%11s:%7.3f%%\n' % (categories_landslide[i], best_score[categories_landslide[i]]))
                    elif args.num_classes == 6:
                        if i % 2 != 1:
                            f.write('%11s:%7.3f%%\t' % (categories_ISPRS[i], best_score[categories_ISPRS[i]]))
                        else:
                            f.write('%11s:%7.3f%%\n' % (categories_ISPRS[i], best_score[categories_ISPRS[i]]))
                    else:
                        if i % 2 != 1:
                            f.write('%11s:%7.3f%%\t' % (categories_coco[i], best_score[categories_coco[i]]))
                        else:
                            f.write('%11s:%7.3f%%\n' % (categories_coco[i], best_score[categories_coco[i]]))
                f.write('\n======================================================\n')
                f.write('%11s:%7.3f%%' % ('mIoU', max_mIoU))
                f.write(f"\nPrediction: {best_score['Prediction']}, Recall: {best_score['Recall']}\n")
