
# MCTformer
The pytorch code for our CVPR2022 paper [Multi-class Token Transformer for Weakly Supervised Semantic Segmentation](https://arxiv.org/abs/2203.02891).

[[Paper]](https://arxiv.org/abs/2203.02891) [[Project Page]](https://xulianuwa.github.io/MCTformer-project-page/)

<p align="center">
  <img src="MCTformer-V1.png" width="720" title="Overview of MCTformer-V1" >
</p>
<p align = "center">
Fig.1 - Overview of MCTformer
</p>



# FRD: Feature Representation Discrepancy for WSSS (on top of MCTformer)

This repository contains our FRD implementation and utilities to:
- generate CAMs,
- search the best CAM threshold on training set,
- export CRF-refined pseudo labels for training a downstream segmentation model.

> **Note**    
> If you need to train or reproduce vanilla MCTformer, please refer to the original project (see “References”).

## Prerequisite
- Ubuntu 20.04, with Python 3.10.13 and the following python dependencies.
```bash
pip install -r requirements.txt
```
- Download [the PASCAL VOC 2012 development kit](http://host.robots.ox.ac.uk/pascal/VOC/voc2012).



# FRD step

## Complete folder format

```text

VOC2012_org/
├─ all_data.txt
├─ cls_labels.txt
├─ train_aug_id.txt
├─ select_image.txt
├─ train_id.txt
├─ val_id.txt
└─ voc12/
   └─ VOCdevkit/
      └─ VOC2012/
         ├─ ImageSets/
         ├─ JPEGImages/
         ├─ SegmentationClass/
         ├─ SegmentationClassAug/
         └─ SlopeImages/
```
#### Pesudo Seed Result
#### PASCAL VOC 2012 dataset
<table>
  <thead>
    <tr>
      <th style="text-align:center;">Model</th>
      <th style="text-align:center;">Backbone</th>
      <th style="text-align:center;">mIoU</th>
      <th style="text-align:center;">Weights</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align:center;">MCTformer-V2</td>
      <td style="text-align:center;">DeiT-small</td>
      <td style="text-align:center;">64.915%</td>
      <td style="text-align:center;"><a href="https://drive.google.com/file/d/1loK45CexEmkilebWFlDUp3zACH0-F6vr/view?usp=sharing">Google drive</a></td>
    </tr>
  </tbody>
</table>

### Generate attention maps
```text
python main.py --model deit_small_MCTformerV2_patch16_224 
--data-set VOC12MS --img-list train_id.txt \
--data-path ./dataset/VOC2012_org --output_dir ./result_dir/MCTformer_results/VOC2012_org \
--resume ./ckpt/FRD_VOC_checkpoint.pth --gen_attention_maps --attention-type fused \
--layer-index 12 --cam-npy-dir ./result_dir/MCTformer_results/VOC2012_org/attn-patchrefine-npy-ms
```

>--data-path The dataset path  
>--img-list Here train_id.txt is used to generate attention maps  
>--output_dir  Output path  
>--resume   load the trained checkpoint  
>--cam-npy-dir Generated attention maps path  
>--gen_attention_maps  Enable CAM generation mode (no training; use --resume; outputs to --cam-npy-dir)  
>--attention-type  CAM type. 'fused' = class-token + patch affinity (default)  
>--layer-index   Transformer block index for CAM (e.g., 12 = last layer; default 12)  


### Verify the results
```text
python evaluation.py --list train_id.txt --data-set VOC12 --data-path ./dataset/VOC2012_org 
--type npy --predict_dir ./result_dir/MCTformer_results/VOC2012_org/attn-patchrefine-npy-ms \
--curve True --start 38 --comment eval_result
```

>--data-path The dataset path  
>--predict_dir Please use the same path as the --cam-npy-dir in the previous command.  
>--start    Threshold start  
>--curve    Sweep thresholds and report the best mIoU（True/False）  
>--comment   A tag written to eval record/log（e.g.eval_result）  


## After verification, please change the following --t parameter to the optimal threshold output after running the above results
Example If it is 0.59, please fill in -- 59

### Generate pseudo label 
```text
python evaluation.py --list train_id.txt --data-set VOC12 --data-path ./dataset/VOC2012_org 
--type npy --predict_dir ./result_dir/MCTformer_results/VOC2012_org/attn-patchrefine-npy-ms \
--t 59 --out-dir ./result_dir/MCTformer_results/VOC2012_org/pseudo-mask-ms-crf --out-crf 
```

>--predict_dir Please use the same path as the --predict_dir in the previous command.  
>--out-dir For the pseudo label path, please put it in the same folder, for example, put it in ./result_dir/MCTformer_results/FRD_20220311_6m/. In this folder, the pseudo-mask-ms-crf is here
