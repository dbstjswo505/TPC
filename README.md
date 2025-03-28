# TPC: Test-time Procrustes Calibration for Diffusion-based Human Image Animation, NeurIPS 2024

[![arXiv](https://img.shields.io/badge/arXiv-TPC-b31b1b.svg)](https://arxiv.org/html/2410.24037v1) 



**TPC** is a framework that enhances the fidelity of image animation by effectively enhancing the robustness under compositional misaligned scenario between target and reference images in a model-agnostic manner.

[//]: # (### Abstract)
>Human image animation aims to generate a human motion video from the inputs of a reference human image and a target motion video. Current diffusion-based image animation systems exhibit high precision in transferring human identity into targeted motion, yet they still exhibit irregular quality in their outputs. Their optimal precision is achieved only when the physical compositions (i.e., scale and rotation) of the human shapes in the reference image and target pose frame are aligned. In the absence of such alignment, there is a noticeable decline in fidelity and consistency. Especially, in real-world environments, this compositional misalignment commonly occurs, posing significant challenges to the practical usage of current systems. To this end, we propose Test-time Procrustes Calibration (TPC), which enhances the robustness of diffusion-based image animation systems by maintaining optimal performance even when faced with compositional misalignment, effectively addressing real-world scenarios. The TPC provides a calibrated reference image for the diffusion model, enhancing its capability to understand the correspondence between human shapes in the reference and target images. Our method is simple and can be applied to any diffusion-based image animation system in a model-agnostic manner, improving the effectiveness at test time without additional training.

## Environment for TPC
```
conda create -n openmmlab_my python=3.8
conda activate openmmlab_my
pip install -r requirements.txt
```
## If it is not working, try below
```
conda create -n openmmlab_my python=3.8
conda activate openmmlab_my
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2

2. (https://detectron2.readthedocs.io/en/latest/tutorials/install.html) SAM
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2

3. (https://mmpose.readthedocs.io/en/latest/installation.html) Openmm
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.1"
mim install "mmdet>=3.1.0"
mim install "mmpose>=1.1.0"

```

## TPC: Prepare Calibrated images for Human Image Animation model
Input: 
 - sample1.PNG (reference image)
 - sample1.mp4 (driving video)

Output:
 - results/cal_image/pose_sample1_ref_sample1
   - 0-person_cal_ref_img
   - 0-person_cal_ref_mask_img
   - 0-person_pose_mask_img
   - 0-person_ref_mask_img
   - pose_pad_mask_img
   - poes_video
   - ref_img

```
cd TPC
python run.py
```

## Environment for Human Image Animation model (e.g., MagicAnimate)
[magicanimate](https://github.com/magic-research/magic-animate)

## Inference Human Image Animation model with TPC
### move calibrated images to Human Image Animation model (e.g., MagicAnimate)
Type target prompt in config/config_sample.yaml to get edited video.
```
cp TPC/TPC/results/cal_image/pose_sample1_ref_sample1 TPC/model/magic-animate/inputs/applications/calibrated_image/
cd model/magicanimate
bash scripts/animate.sh
```

## Acknowledgement

This code is implemented on top of following contributions: [magicanimate](https://github.com/magic-research/magic-animate), [SAM](https://github.com/facebookresearch/segment-anything), [DensePose](https://github.com/facebookresearch/DensePose), [MMPose](https://github.com/open-mmlab/mmpose)

We thank the authors for open-sourcing these great projects and papers!

This work was supported by Institute for Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT) (No.RS-2021-II211381, Development of Causal AI through Video Understanding and Reinforcement Learning, and Its Applications to Real Environments) and partly supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT) (No.RS-2022-II220184, 2022-0-00184, Development and Study of AI Technologies to Inexpensively Conform to Evolving Policy on Ethics)

## Citation
Please kindly cite our paper if you use our code, data, models or results:

```bibtex
@article{yoon2024tpc,
  title={Tpc: Test-time procrustes calibration for diffusion-based human image animation},
  author={Yoon, Sunjae and Koo, Gwanhyeong and Lee, Younghwan and Yoo, Chang},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={118654--118677},
  year={2024}
}
```
