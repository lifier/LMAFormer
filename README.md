## PyTorch implementation of "LMAFormer: Local Motion Aware Transformer for Small Moving Infrared Target Detection" 
[Project](https://github.com/lifier/LMAFormer-master) - [Paper](Link to upload the paper after acceptance)
<hr>

# Abstract
<p align="justify">
In temporal infrared small target detection, it is crucial to leverage the disparities in spatio-temporal characteristics between the target and the background to distinguish the former. However, remote imaging and the relative motion between the detection platform and the background cause significant coupling of spatio-temporal characteristics, making target detection highly challenging. To address these challenges, we propose a network named LMAFormer. First, we introduce a local motion-aware spatio-temporal attention mechanism that leverages motion and temporal energy variations between infrared small targets and backgrounds to extract locally salient spatio-temporal features of targets. Second, we employ a multi-scale fusion transformer encoder that computes self-attention weights across and within scales during encoding, acquiring multi-scale global spatio-temporal information for enhanced motion background modeling. Lastly, we propose a multi-frame joint query decoder. The shallowest feature map after multi-scale feature propagation is mapped to initial query weights, which are refined through grouped convolutions to generate grouped query vectors. These are jointly optimized to encapsulate rich multi-frame details, strengthening motion background modeling and target feature representation, improving prediction accuracy. Experimental results on the NUDT-MIRSDT, IRDST and the established TSIRMT datasets demonstrate that our network outperforms state-of-the-art (SOTA) methods.
</p>



# Architecture
<p align="center">
  <img src="pic/LMAFormer_fig_1.png" width="auto" alt="accessibility text">
</p>
Overall Architecture of LMAFormer.

# Installation


## Environment Setup
The experiments were done on Windows11 with python 3 using anaconda environment. Here is details on how to set up the conda environment.
(If you do not have anaconda 3 installed, first do it following the set up instruction from [here](https://www.anaconda.com/products/distribution)) 

* Create conda environment:
 
  ```create environment
  conda create -n LMAFormer python=3
  conda activate LMAFormer
  ```

* Install PyTorch from [here](https://pytorch.org/get-started/locally/). 


* Install other requirements:

  ```setup
  pip install -r requirements.txt

## Datasets
We evaluate network performance using NUDR-MIRSDT, IRDST and a self-built dataset TSIRMT

Download the datasets following the corresponding paper/project page and update dataset paths in 'datasets/path_config.py'. 
Here is the list of datasets used. 

- [NUDT-MIRSDT](https://mail.nudt.edu.cn/coremail/common/nfFile.jsp?share_link=5418DA3CCDBA48F8A81AE154DA02C3D5&uid=liruojing%40nudt.edu.cn) (Extraction code: M6PZ)
- [IRDST](https://drive.google.com/file/d/1sb-32pydlpXvlNxwx9niT2t6KP9oMJID/view?usp=sharing)
- [TSIRMT](https://drive.google.com/drive/folders/1aWDNdUWkTOuV3fILbgLDEqM2N2erW05n?usp=sharing)

## Download Trained Models 
Pretrained Swin backbones can be downloaded from their corresponding repository. 
- [Swin-S](https://github.com/SwinTransformer/storage/releases/download/v1.0.4/swin_small_patch244_window877_kinetics400_1k.pth)
- [Swin-B](https://github.com/SwinTransformer/storage/releases/download/v1.0.4/swin_base_patch244_window877_kinetics400_22k.pth)
- [Resnet-101 COCO](https://drive.google.com/file/d/1NGuaew1d0x0kdK2XR_J3Vlmc6OGtOe58/view?usp=sharing)
If you are interested in evaluating only, you can download the selected trained LMAFormer checkpoints from the links in the results table.   


# Training 

The models were trained and tested using a single NVIDIA 4080 GPU.  

* Train LMAFormer with Swin backbone on NUDT-MIRSDR, IRDST, TSIRMT datasets:
  ```
  python train_swin.py
  ```

# Inference
### Inference on NUDT-MIRSDT:
    ```
        python inference_swin.py  --model_path ./result/TSIRMT/checkpoint_NUDT-MIRSDT.pth  --dataset NUDT-MIRSDT --val_size 400 --flip --msc --output_dir ./predict/NUDT-MIRSDT  
    ```
    Expected miou: 73.26
### Inference on IRDST:
    ```
        python inference_swin.py  --model_path ./result/IRDST/checkpoint_IRDST.pth  --dataset IRDST --val_size 400 --flip --msc --output_dir ./predict/IRDST
    ```
    Expected miou: 59.17
### Inference on TSIRMT:
    ```
        python inference_swin.py  --model_path ./result/TSIRMT/checkpoint_TSIRMT.pth  --dataset TSIRMT --val_size 400 --flip --msc  --output_dir ./predict/TSIRMT
    ```
    Expected miou: 65.89

## Results Summary
### Results on NUDT-MIRSDT, IRDST and TSIRMT
| Dataset  | Checkpoint                                                                                        | IoU  | nIoU | Pd | Fa |
|-----------|---------------------------------------------------------------------------------------------------|------|------|------|------|
| NUDT-MIRSDT | [checkpoint](https://drive.google.com/drive/folders/0ABYEIdgnW9YvUk9PVA)  | 73.26 | 73.63 | 99.68 | 0.71 |
| IRDST | [checkpoint](https://drive.google.com/drive/folders/0ABYEIdgnW9YvUk9PVA)  | 59.17 | 57.51 | 99.64 | 14.95 |
| TSIRMT | [checkpoint](https://drive.google.com/drive/folders/0ABYEIdgnW9YvUk9PVA)  | 65.89 | 65.63 | 86.10 | 185.78 |

### Acknowledgement
We would like to thank the open-source projects with  special thanks to [DETR](https://github.com/facebookresearch/detr)  and [VisTR](https://github.com/Epiphqny/VisTR) for making their code public. Part of the code in our project are collected and modified from several open source repositories.

## Citation
Please consider citing our paper in your publications if the project helps your research. BibTeX reference is as follow.

```
@inproceedings{LMAFormer24,
  title={ {LMAFormer}: Local Motion Aware Transformer for Small Moving Infrared Target Detection},
  author={Yuanxin Huang, Xiyang Zhi, JianmingHu, Lijian Yu, Qichao Han, Wenbin Chen, Wei Zhang},
  booktitle={IEEE Transactions on Geoscience and Remote Sensing},
  year={2024}
}
```

