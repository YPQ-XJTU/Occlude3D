# Occlude3D:Make Occlusion Clear in Monocular 3D Object Detection

![](C:\Users\13908\Desktop\Occulde3D\Imgs\Fig1.png)

## Abstract

Monocular 3D object detection is getting increasingly popular in autonomous driving research owing to its cost and time efficiency. Current methods rely on detecting independent targets, while accuracy declines notably when occlusion occurs, which is a critical and urgent challenge. Relying solely on local features of a single object cannot capture the complete 3D properties of objects in a scene, as it ignores the implicit spatial relationships between different instances. 

To alleviate this problem, we propose a **Center-guided Depth-aware** strategy to capture the positional relationships among instances, called Occlude3D.  Specifically, we design a Center Feature Fusion Module (CFM)  to process the input visual feature and depth feature into learnable queries, and we also propose an Occlusion-Aware Decoder that infers implicit spatial constraints between objects through deep interaction between central visual features and depth features. On the KITTI benchmark, our method significantly increases the state-of-the-art methods by **9.7\%** and **15.2\%** in the moderate and the hard categories. Besides, our method demonstrates significant performance on the Waymo dataset as well.

![](\Imgs\Fig2.png)

## Main result
### KITTI Benchmark

<table>
    <thead>
        <tr>
            <th rowspan="2" style="text-align:center;">Approaches</th>
            <th rowspan="2" style="text-align:center;">Venue</th>
            <th rowspan="2" style="text-align:center;">Modality</th>
            <th colspan="3" style="text-align:center;">Test AP<sub>BEV|R40</sub></th>
            <th colspan="3" style="text-align:center;">Test AP<sub>3D|R40</sub></th>
            <th colspan="3" style="text-align:center;">Val AP<sub>BEV|R40</sub></th>
        </tr>
        <tr>
            <th style="text-align:center;">Easy</th>
            <th style="text-align:center;">Mod</th>
            <th style="text-align:center;">Hard</th>
            <th style="text-align:center;">Easy</th>
            <th style="text-align:center;">Mod</th>
            <th style="text-align:center;">Hard</th>
            <th style="text-align:center;">Easy</th>
            <th style="text-align:center;">Mod</th>
            <th style="text-align:center;">Hard</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td style="text-align:center;">MonoDETR</td>
            <td style="text-align:center;"><i>ICCV23</i></td>
            <td style="text-align:center;">VD</td>
            <td style="text-align:center;">32.20</td>
            <td style="text-align:center;">21.45</td>
            <td style="text-align:center;">18.68</td>
            <td style="text-align:center;">24.52</td>
            <td style="text-align:center;">16.26</td>
            <td style="text-align:center;">13.93</td>
            <td style="text-align:center;">28.84</td>
            <td style="text-align:center;">20.61</td>
            <td style="text-align:center;">16.38</td>
        </tr>
        <tr>
            <td style="text-align:center;">MonoCD</td>
            <td style="text-align:center;"><i>CVPR24</i></td>
            <td style="text-align:center;">VD</td>
            <td style="text-align:center;">33.41</td>
            <td style="text-align:center;">22.81</td>
            <td style="text-align:center;">19.57</td>
            <td style="text-align:center;">25.53</td>
            <td style="text-align:center;">16.59</td>
            <td style="text-align:center;">14.53</td>
            <td style="text-align:center;">28.34</td>
            <td style="text-align:center;">20.85</td>
            <td style="text-align:center;">16.13</td>
        </tr>
        <tr>
            <td rowspan="2" style="text-align:center;"><b>Occlude3D (Ours)</b></td>
            <td style="text-align:center;"><i>None</i></td>
            <td style="text-align:center;">VD</td>
            <td style="text-align:center;"><b>34.15</b></td>
            <td style="text-align:center;"><b>25.41</b></td>
            <td style="text-align:center;"><b>22.07</b></td>
            <td style="text-align:center;"><b>27.05</b></td>
            <td style="text-align:center;"><b>18.20</b></td>
            <td style="text-align:center;"><b>16.75</b></td>
            <td style="text-align:center;"><b>30.76</b></td>
            <td style="text-align:center;"><b>23.15</b></td>
            <td style="text-align:center;"><b>19.23</b></td>
        </tr>
        <tr>
            <td colspan="2" style="text-align:center;"><i>Improvement v.s. second-best</i></td>
            <td style="text-align:center;">+0.74</td>
            <td style="text-align:center;">+3.40</td>
            <td style="text-align:center;">+2.50</td>
            <td style="text-align:center;">+1.52</td>
            <td style="text-align:center;">+1.61</td>
            <td style="text-align:center;">+2.22</td>
            <td style="text-align:center;">+2.32</td>
            <td style="text-align:center;">+2.30</td>
            <td style="text-align:center;">+3.10</td>
    </tr>

    </tbody>
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
