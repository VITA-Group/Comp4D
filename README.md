# Comp4D: LLM-Guided Compositional 4D Scene Generation

Authors: Dejia Xu, Hanwen Liang, Neel P. Bhatt, Hezhen Hu, Hanxue Liang,
Konstantinos N. Plataniotis, and Zhangyang Wang

[[Project Page]](https://vita-group.github.io/Comp4D/) | [[Video (narrated)]](https://www.youtube.com/watch?v=9q8SV1Xf_Xw) | [[Video (results)]](https://www.youtube.com/watch?v=gXVoPTGb734) | [[Paper]](https://github.com/VITA-Group/Comp4D/blob/main/assets/Comp4D.pdf) | [[Arxiv]](https://arxiv.org/abs/2403.16993)

## News

- 2024.4.1:  Released code!
- 2024.3.25:  Released on arxiv!

## Overview

![overview](docs/static/media/task.29476c66b38120ba3c46.jpg)

As show in figure above, we introduce **Comp**ositional **4D** Scene Generation. Previous works concentrate on object-centric 4D objects with limited movement. In comparison, our work extends the boundaries to the demanding task of compositional 4D scene generation. We integrate GPT-4 to decompose the scene and design proper trajectories, resulting in larger-scale movements and more realistic object interactions.

## Representative Results

<table class="center">
  <td><video src="https://github.com/VITA-Group/Comp4D/blob/main/assets/butterfly_flower1.mp4" width="170"></video>
  <td><video src="https://github.com/VITA-Group/Comp4D/blob/main/assets/butterfly_flower2.mp4" width="170"></video>
  <tr>
  <tr>
  <td style="text-align:center;" width="170">"a butterfly flies towards the flower"</td>
  <td style="text-align:center;" width="170">"a girl is riding a horse fast on grassland"</td>

</table >

## Static Assets

We release our pre-generated static assets in `data/` directory. During training we keep the static 3D Gaussians fixed and only optimize the deformation modules.

## Setup
```bash
conda env create -f environment.yml
conda activate Comp4D
pip install -r requirements.txt

# 3D Gaussian Splatting modules, skip if you already installed them
# a modified gaussian splatting (+ depth, alpha rendering)
git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization
pip install ./diff-gaussian-rasterization
pip install ./simple-knn
```

## Example Case
#### Prompt Case
"a butterfly flies towards the flower"

#### Compositional Scene training
```
python train_comp.py --configs arguments/comp_butterfly_flower_zs.py -e butterflyflower_exp --image_weight_override 0.02 --nn_weight 1000 --with_reg --cfg_override 100.0 --loss_dx_weight_override 0.005
```

#### Rendering
```
python render_comp_video.py --skip_train --configs arguments/comp_butterfly_flower_zs.py --skip_test --model_path output_demo/date/butterflyflower_exp_date/ --iteration 3000
```

## Citation

If you find this repository/work helpful in your research, please consider citing the paper and starring the repo ⭐.
```
@article{xu2024comp4d,
  title={Comp4D: LLM-Guided Compositional 4D Scene Generation},
  author={Xu, Dejia and Liang, Hanwen and Bhatt, Neel P and Hu, Hezhen and Liang, Hanxue and Plataniotis, Konstantinos N and Wang, Zhangyang},
  journal={arXiv preprint arXiv:2403.16993},
  year={2024}
}
```
