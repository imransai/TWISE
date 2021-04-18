# Twin-Surface-Extrapolation-TWISE-
Code for CVPR 2021 Submission of our paper 'Depth Completion with Twin Surface Extrapolation at Occlusion Boundaries' available at
[paper.](https://arxiv.org/abs/2104.02253)

# Overview
The following is a teaser result of our proposed algorithm:
![Image](/Images/twise_teaser.png)
The figure describes how our depth completion algorithm can input LiDAR data and image (a), and extrapolate the estimates of foreground depth $d_1$ (b) and background depth $d_2$ (c), along with a weight $\sigma$ (e). Fusing all three leads to the completed depth (d). The foreground-background depth difference (f) $d_2-d_1$ is small except at depth discontinuities.
# Dependencies
We built our framework on Pytorch 1.3 and CUDA 10.1 in Ubuntu 18.1 and python 3.6. In terminal, the steps are:
1. conda create -n pytorch1.3_twise python=3.6
2. source activate pytorch1.3_twise
3. conda install pytorch==1.3.0 torchvision==0.4.1 cudatoolkit=10.1 -c pytorch

# Training
We will release our training code soon. Stay tuned!

# PreTrained Model
Our best pre-trained model on KITTI is obtained with Multi-Stack HourGlass Network. Our best pre-trained model can be obtained from [here](https://drive.google.com/file/d/1rSNi_XqXQuDtPb9fVzIf8z1liv4mynL_/view?usp=sharing) 


# Evaluation
We also provide an evaluation script to evaluate the model on 1000 validation images of KITTI. Download the selected images folder from [KITTI's website] (http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion) and extract it in Data folder. The Data folder should look like this:
Data
  |__depth_selection
  
  To evaluate the model, just extract the model provided in tar.gz format in the pretrained_models dir and run `python evaluate.py`. It evaluates the model based the metrics MAE, RMSE, TMAE, TRMSE etc.

# Virtual KITTI 
## Modelling outliers with Semi-Dense GT

VKITTI Sparse Lidar| KITTI 64R Lidar Sparse
---  | ---
(a)  |  (b)
![](Images/vkitti_sparsesample.JPG) | ![](Images/kitti_sparsesample.JPG)

As we claim our method works well on boundaries, we also evaluate on VKITTI 2.0, a synthetic dataset with clean and dense GT depth at depth discontinuities. The VKITTI $2.0$, created by the Unity game engine, contains $5$ different camera locations ($15^o$ left, $15^o$ right, $30^o$ left, $30^o$ right, clone) in addition to $5$ different driving sequences. Additionally, there are stereo image pairs for each camera location. For training and testing, we only use the clone (forward facing camera) with stereo image pairs. For VKITTI training, $2$k training images were created from driving sequences $01$, $02$, $06$, and $018$ respectively. For testing, we use sequence $020$ at the left stereo camera, and choose every other frames, with total $420$ images.
We subsample the dense GT depth in azimuth-elevation space to simulate LiDAR-like pattern as sparse inputs. 
Further, we create the pseudo GT following~\cite{uhrig2017sparsity} to study the effects of outlier noise on training and evaluation.
 
In the main paper, we performed an ablation study on Virtual KITTI \cite{cabon2020virtual} (VKITTI) using semi-dense and sparse samples created from dense VKITTI depth maps. We created semi-dense VKITTI to simulate outlier noise similar to that existing in real KITTI dataset. In this section, we discuss the data generation process in detail and show some visual examples of how the sparse depth/semi-dense compares with sparse/semi-dense gt of KITTI dataset in Fig..

The dense ground-truth depth maps from VKITTI contains accurate depth on object discontinuities. Using this as a reference, we subsampled the ground-truth depth maps. Instead of uniformly subsampling the GT depth, we subsample the LiDAR in the azimuth-elevation coordinates to make the input sparse depth resemble structured patterns found in original LiDAR (see ($a$) and ($b$) of Fig. \ref{fig:vkitti_okitti}). The subsampled depth from the left camera is then projected to the right camera, and vice versa to simulate LiDAR points projected onto images in real-world scenes. For supervision, GT depth beyond $90$m are suppressed to simulate LiDAR points with no returns (see ($e$) of Fig.~\ref{fig:vkitti_okitti}). 
In addition to supervision using clean ground-truth present, we also perform supervision on Semi-Dense GT of VKITTI (Fig.~$10$ of the main paper) created by simulating outliers existing in original KITTI dataset \cite{uhrig2017sparsity}. In the KITTI dataset,  semi-dense GT is created by accumulating LiDAR points from $+/-5$ frames from the reference frame. 
We follow the similar procedure as followed by Uhrig et al when creating semi-dense GT. Refer to Fig. for a comparison between semi-dense VKITTI and semi-dense KITTI (see ($c$) and ($d$) of Fig..

# Citations
If you use our method and code in your work, please cite the following:

@inproceedings{ depth-completion-with-twin-surface-extrapolation-at-occlusion-boundaries,
  author = { Saif Imran and Xiaoming Liu and Daniel Morris },
  title = { Depth Completion with Twin-Surface Extrapolation at Occlusion Boundaries },
  booktitle = { In Proceeding of IEEE Computer Vision and Pattern Recognition },
  address = { Nashville, TN },
  month = { June },
  year = { 2021 },
}
