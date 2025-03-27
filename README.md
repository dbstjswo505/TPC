# FRAG: Frequency Adaptive Group for Diffusion Video Editing, ICML 2024
## [<a href="https://dbstjswo505.github.io/FRAG-page/" target="_blank">Project Page</a>]

[![arXiv](https://img.shields.io/badge/arXiv-TPC-b31b1b.svg)](https://arxiv.org/html/2410.24037v1) 

[![Video Label](http://img.youtube.com/vi/9XPNElLv3Y4/0.jpg)](https://youtu.be/9XPNElLv3Y4)


**FRAG** is a framework that enhances the quality of edited videos by effectively preserving high-frequency components.

[//]: # (### Abstract)
>Human image animation aims to generate a human motion video from the inputs of a reference human image and a target motion video. Current diffusion-based image animation systems exhibit high precision in transferring human identity into targeted motion, yet they still exhibit irregular quality in their outputs. Their optimal precision is achieved only when the physical compositions (i.e., scale and rotation) of the human shapes in the reference image and target pose frame are aligned. In the absence of such alignment, there is a noticeable decline in fidelity and consistency. Especially, in real-world environments, this compositional misalignment commonly occurs, posing significant challenges to the practical usage of current systems. To this end, we propose Test-time Procrustes Calibration (TPC), which enhances the robustness of diffusion-based image animation systems by maintaining optimal performance even when faced with compositional misalignment, effectively addressing real-world scenarios. The TPC provides a calibrated reference image for the diffusion model, enhancing its capability to understand the correspondence between human shapes in the reference and target images. Our method is simple and can be applied to any diffusion-based image animation system in a model-agnostic manner, improving the effectiveness at test time without additional training.

## Environment
```
conda create -n frag python=3.9
conda activate frag
pip install -r requirements.txt
```
## DDIM inversion
Type source prompt in config/config_sample.yaml to get ddim latent features.
```
python ddim_inversion.py
```
## Editing
Type target prompt in config/config_sample.yaml to get edited video.
```
python frag.py
```

## Acknowledgement

This code is implemented on top of following contributions: [TAV](https://github.com/showlab/Tune-A-Video), [TokenFlow](https://github.com/omerbt/TokenFlow), [HuggingFace](https://github.com/huggingface/transformers), [FLATTEN](https://github.com/yrcong/flatten), [FateZero](https://github.com/ChenyangQiQi/FateZero), [Prompt-to-prompt](https://github.com/google/prompt-to-prompt) 

We thank the authors for open-sourcing these great projects and papers!

This work was supported by Institute for Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT) (No. 2021-0-01381, Development of Causal AI through Video Understanding and Reinforcement Learning, and Its Applications to Real Environments) and partly supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT) (No.2022-0-00184, Development and Study of AI Technologies to Inexpensively Conform to Evolving Policy on Ethics).

## Citation
Please kindly cite our paper if you use our code, data, models or results:

```bibtex
@article{yoon2024frag,
  title={FRAG: Frequency Adapting Group for Diffusion Video Editing},
  author={Yoon, Sunjae and Koo, Gwanhyeong and Kim, Geonwoo and Yoo, Chang D},
  journal={arXiv preprint arXiv:2406.06044},
  year={2024}
}
```
