# Occlude3D:Make Occlusion Clear in Monocular 3D Object Detection

![](C:\Users\13908\Desktop\Occulde3D\Imgs\Fig1.png)

## Abstract

Monocular 3D object detection is increasingly popular in autonomous driving research owing to its cost and time efficiency. The main focus of current research is on independently detecting targets, while accuracy declines notably when occluded targets are present, which is a critical and urgent challenge. It is insufficient to rely solely on the local traits of individual objects to offer a holistic depiction of the scene-level 3D object attributes, as it ignores the implicit spatial relationships among diverse instances.

We propose a strategy called **Center-guided Depth-aware** to capture the **implicit positional relationships among instances**, called Occlude3D. Specifically, we design the Center-Feature Fusion Module (CFM) and Decoupled Depth-Feature Module (DDFM) to process the input visual feature and depth feature into learnable queries, and we utilize a Center-guided decoder to facilitate depth interaction between objects and scene information, thus deriving implicit spatial constraints among scene targets. On KITTI benchmark with monocular images as input, Occlude3D achieves state-of-the-art performance. Compared to several transformer-based models, it attains peak performance with a reduced number of model parameters and superior operational efficiency.

![]\Imgs\Fig2.png)

## Main result

<table>
    <tr>
        <td rowspan="2",div align="center">Models</td>
        <td colspan="3",div align="center">Val, AP<sub>3D|R40</sub></td>   
    </tr>
    <tr>
        <td div align="center">Easy</td> 
        <td div align="center">Mod.</td> 
        <td div align="center">Hard</td> 
    </tr>
    <tr>
        <td rowspan="4",div align="center">Occlude3D</td>
        <td div align="center">31.25%</td> 
        <td div align="center">23.36%</td> 
        <td div align="center">18.70%</td> 
    </tr>  
</table>



## Installation

1. create a conda environment:

```bash
conda create -n occlude3d python=3.8 -y
conda activate occlude3d
```

2. Install pytorch and torchvision matching your CUDA version:

```bash
conda install pytorch torchvision cudatoolkit
# We adopt torch 1.10.0+cu111
```

3. Install requirements and compile the deformable attention:

```bash
pip install -r requirements.txt

cd lib/models/occlude3d/ops/
bash make.sh

cd ../../../..
```

4. Download KITTI datasets and prepare the directory structure as:

```
│Occlude3D/
├── data
│   │── KITTI3D
|   │   │── training
|   │   │   ├──calib & label_2 & image_2 & depth_dense
|   │   │── testing
|   │   │   ├──calib & image_2
├──...
```



## Get Started

### Train

You can modify the settings of models and training in `configs/occlude3d.yaml` and indicate the GPU in `train.sh`:

    bash train.sh configs/occlude3d.yaml 

### Test

The best checkpoint will be evaluated as default. You can change it at "./checkpoint" in `configs/occlude3d.yaml`:

```
bash test.sh configs/occlude3d.yaml
```



## Some Qualitative results

![](C:\Users\13908\Desktop\Occulde3D\Imgs\Fig3.png)



![](C:\Users\13908\Desktop\Occulde3D\Imgs\Fig4.png)
