
# References - MCTformer
The pytorch code for our CVPR2022 paper [Multi-class Token Transformer for Weakly Supervised Semantic Segmentation](https://arxiv.org/abs/2203.02891).

[[Paper]](https://arxiv.org/abs/2203.02891) [[Project Page]](https://xulianuwa.github.io/MCTformer-project-page/)
> If you need to train or reproduce vanilla MCTformer, please rely on the original repo’s docs and training scripts.

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

### Train (FRD)
```text
python main.py \
--model deit_small_MCTformerV2_patch16_224 \
--batch-size 128 --data-set VOC12 --img-list train_id.txt \
--data-path ./dataset/WSSS_dataset/VOC2012_org \
--output_dir /result_dir/MCTformer_results/FRD_b128_w10e2_th49 \
--pin-mem --epochs 5 --layer-index 12 --warmup-epochs 0 \
--warmup-lr 0.0001 --lr 0.0001 --min-lr 0.0001 \
--finetune ckpt/MCTformer_offical_checkpoint.pth \
--use-prototypes --prototypes_weight 0.12 --mask_thresh 0.49 \
--no-if_eval_miou
```

>--data-path          VOC dataset root (contains voc12/VOCdevkit/VOC2012/…)  
>--img-list           Training image ID list (e.g., train_id.txt / train_aug_id.txt)  
>--output_dir         Where logs/checkpoints are saved  
>--finetune           Init weights (official MCTformer checkpoint)  
>--use-prototypes     Enable FRD (prototype loss)  
>--prototypes_weight  Weight of FRD loss (e.g., 0.12)  
>--mask_thresh        CAM foreground threshold used by FRD (e.g., 0.49)  
>--no-if_eval_miou    Disable on-the-fly mIoU eval during training (faster)  
>--batch-size / --epochs / --lr / --warmup-* / --min-lr  Usual training knobs  
>--layer-index        Attention layer used internally for CAM cues (keep 12)  

#### Notes

- For full training, set `--epochs` to your official schedule (the example 5 is only for a quick run) and adjust the learning rate as needed.
- After training, `--output_dir` will contain `checkpoint.pth` and `checkpoint_best_patch.pth`; when generating CAMs (Step 1), use `checkpoint_best_patch.pth` as `--resume`.
- If your dataset path is `./dataset/WSSS_dataset/VOC2012_org`, update `--data-path` accordingly.


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
