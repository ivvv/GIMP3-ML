# GIMP3-ML

Machine Learning plugins for GIMP 3.

Forked from the [original version](https://github.com/kritiksoman/GIMP-ML/tree/GIMP3-ML) to improve the user experience in several aspects:
* Added more models.
* Models are run with Python 3.10+.
* Full error text is shown in the GIMP error dailog and in debug console.
* Additional alpha channel handling in some plugins.
* Automatic installation for Windows systems.
* And other smaller improvements.

The plugins have been tested with GIMP 2.99.12 on the following systems: <br>
* Windows 10

# Installation Steps
1. Install [GIMP3](https://www.gimp.org/downloads/devel/).
2. Download this repository.
3. On Windows:
      * Install [Python 3.10](https://www.python.org/downloads/).
      * Run `install.cmd` from the unpacked folder.
4. You should now find the GIMP-ML plugins under Layers → GIMP-ML. 
5. You can download [the weights here](https://drive.google.com/drive/folders/1ko7j1WOJltJcv-goIBNTIGGniZ68kEQa), or from the weight links below.

![Screenshot](screenshot.png)

# References
### Background Removal
* Source: https://github.com/danielgatis/rembg
* Weights: 
    - u2net ([download](https://drive.google.com/uc?id=1tCU5MM1LhRgGou5OpmpjBQbSrYIUoYab), [source](https://github.com/xuebinqin/U-2-Net)): A pre-trained model for general use cases.
    - u2netp ([download](https://drive.google.com/uc?id=1tNuFmLv0TSNDjYIkjEdeH1IWKQdUA4HR), [source](https://github.com/xuebinqin/U-2-Net)): A lightweight version of u2net model.
    - u2net_human_seg ([download](https://drive.google.com/uc?id=1ZfqwVxu-1XWC1xU1GHIP-FM_Knd_AX5j), [source](https://github.com/xuebinqin/U-2-Net)): A pre-trained model for human segmentation.
    - *(unused) u2net_cloth_seg* ([download](https://drive.google.com/uc?id=15rKbQSXQzrKCQurUjZFg8HqzZad8bcyz), [source](https://github.com/levindabhi/cloth-segmentation)): A pre-trained model for Cloths Parsing from human portrait. Here clothes are parsed into 3 category: Upper body, Lower body and Full body.
* License: MIT License

### Anime-style Inpainting
* Source: https://github.com/youyuge34/Anime-InPainting
* Weights: [Google Drive](https://drive.google.com/file/d/12I-K7GQEXEL_rEOVJnRv7ecVHyuZE-1-/view?usp=sharing) | [Baidu](https://pan.baidu.com/s/1WkeRtYViGGGw4fUqPo3nsg)
* License: [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/)
```
@inproceedings{nazeri2019edgeconnect,
  title={EdgeConnect: Generative Image Inpainting with Adversarial Edge Learning},
  author={Nazeri, Kamyar and Ng, Eric and Joseph, Tony and Qureshi, Faisal and Ebrahimi, Mehran},
  journal={arXiv preprint},
  year={2019}}
```
### Demosaics
* Source: 
  * Demosaics: https://github.com/rekaXua/demosaic_project
  * ESRGAN: https://github.com/xinntao/ESRGAN
* Weights: [4x_FatalPixels](https://de-next.owncube.com/index.php/s/mDGmi7NgdyyQRXL/download?path=%2F&files=4x_FatalPixels_340000_G.pth)
* Licenses: 
  * Demosaics: GNU Affero General Public License v3.0
  * ESRGAN: Apache-2.0 license 
> [[Paper](https://arxiv.org/abs/2107.10833)] <br>
> [Xintao Wang](https://xinntao.github.io/), Liangbin Xie, [Chao Dong](https://scholar.google.com.hk/citations?user=OSDCB0UAAAAJ), [Ying Shan](https://scholar.google.com/citations?user=4oXBp9UAAAAJ&hl=en) <br>
> Applied Research Center (ARC), Tencent PCG<br>
> Shenzhen Institutes of Advanced Technology, Chinese Academy of Sciences

### Inpainting
* Source: https://github.com/a-mos/High_Resolution_Image_Inpainting
* License: [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/)
```
@article{Moskalenko_2020,
	doi = {10.51130/graphicon-2020-2-4-18},
	url = {https://doi.org/10.51130%2Fgraphicon-2020-2-4-18},
	year = 2020,
	month = {dec},
	pages = {short18--1--short18--9},
	author = {Andrey Moskalenko and Mikhail Erofeev and Dmitriy Vatolin},
	title = {Deep Two-Stage High-Resolution Image Inpainting},
	journal = {Proceedings of the 30th International Conference on Computer Graphics and Machine Vision ({GraphiCon} 2020). Part 2}} 
```
### SRResNet
* Source: https://github.com/twtygqyy/pytorch-SRResNet
* Torch Hub fork: https://github.com/valgur/pytorch-SRResNet
* License: [MIT](https://github.com/twtygqyy/pytorch-SRResNet/blob/master/LICENSE)
* C. Ledig et al., “[Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](http://arxiv.org/abs/1609.04802),”
  in *2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2017, pp. 105–114.

### Neural Colorization
* Source: https://github.com/zeruniverse/neural-colorization
* Torch Hub fork: https://github.com/valgur/neural-colorization
* License:
   * [GNU GPL 3.0](https://github.com/zeruniverse/neural-colorization/blob/pytorch/LICENSE) for personal or research use
   * Commercial use prohibited
   * Model weights released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
* Based on fast-neural-style:
   * https://github.com/jcjohnson/fast-neural-style
   * License:
      * Free for personal or research use
      * For commercial use please contact the authors
   * J. Johnson, A. Alahi, and L. Fei-Fei, “[Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://cs.stanford.edu/people/jcjohns/papers/eccv16/JohnsonECCV16.pdf),”
     in *Lecture Notes in Computer Science (including subseries Lecture Notes in Artificial Intelligence and Lecture Notes in Bioinformatics)*,
     vol. 9906 LNCS, 2016, pp. 694–711.

### Edge Detection (DexiNed)
* Source: https://github.com/xavysp/DexiNed
* Weights: [BIPED](https://drive.google.com/file/d/1V56vGTsu7GYiQouCIKvTWl5UKCZ6yCNu/view?usp=sharing)
* License: MIT license 
```
@misc{soria2021dexined_ext,
    title={Dense Extreme Inception Network for Edge Detection},
    author={Xavier Soria and Angel Sappa and Patricio Humanante and Arash Arbarinia},
    year={2021},
    eprint={arXiv:2112.02250},
    archivePrefix={arXiv},
    primaryClass={cs.CV}}
```
### DeblurGANv2
* Source: https://github.com/TAMU-VITA/DeblurGANv2
* Torch Hub fork: https://github.com/valgur/DeblurGANv2
* License: [BSD 3-clause](https://github.com/TAMU-VITA/DeblurGANv2/blob/master/LICENSE)
* O. Kupyn, T. Martyniuk, J. Wu, and Z. Wang, “[DeblurGAN-v2: Deblurring (Orders-of-Magnitude) Faster and Better](https://arxiv.org/abs/1908.03826),”
  in *2019 IEEE/CVF International Conference on Computer Vision (ICCV)*, 2019, pp. 8877–8886.

### Monodepth2
* Source: https://github.com/nianticlabs/monodepth2
* Torch Hub fork: https://github.com/valgur/monodepth2
* License:
   * See the [license file](https://github.com/nianticlabs/monodepth2/blob/master/LICENSE) for terms
   * Copyright © Niantic, Inc. 2019. Patent Pending. All rights reserved.
   * Non-commercial use only
* C. Godard, O. Mac Aodha, M. Firman, and G. Brostow, “[Digging Into Self-Supervised Monocular Depth Estimation](http://arxiv.org/abs/1806.01260),”
  in *2019 IEEE/CVF International Conference on Computer Vision (ICCV)*, 2019, pp. 3827–3837.

# Authors
* UserUnknownFactor
* Kritik Soman ([kritiksoman](https://github.com/kritiksoman)) – original GIMP-ML implementation

# License
MIT

Please note that additional license terms apply for each individual model. See the [references](#references) list for details.
Many of the models restrict usage to non-commercial or research purposes only.
