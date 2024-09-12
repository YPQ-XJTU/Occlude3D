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
            <td style="text-align:center;">MonoEF</td>
            <td style="text-align:center;"><i>CVPR21</i></td>
            <td style="text-align:center;">V</td>
            <td style="text-align:center;">29.03</td>
            <td style="text-align:center;">19.70</td>
            <td style="text-align:center;">17.26</td>
            <td style="text-align:center;">21.29</td>
            <td style="text-align:center;">13.87</td>
            <td style="text-align:center;">11.71</td>
            <td style="text-align:center;">-</td>
            <td style="text-align:center;">-</td>
            <td style="text-align:center;">-</td>
        </tr>
        <tr>
            <td style="text-align:center;">MonoRCNN</td>
            <td style="text-align:center;"><i>ICCV21</i></td>
            <td style="text-align:center;">V</td>
            <td style="text-align:center;">25.48</td>
            <td style="text-align:center;">18.11</td>
            <td style="text-align:center;">14.10</td>
            <td style="text-align:center;">18.36</td>
            <td style="text-align:center;">12.65</td>
            <td style="text-align:center;">10.03</td>
            <td style="text-align:center;">16.61</td>
            <td style="text-align:center;">13.19</td>
            <td style="text-align:center;">10.65</td>
        </tr>
        <!-- Continue adding rows similarly -->
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
            <td style="text-align:center;" colspan="3"><span style="color:green;">+0.74</span>, <span style="color:green;">+3.40</span>, <span style="color:green;">+2.50</span></td>
            <td style="text-align:center;" colspan="3"><span style="color:green;">+1.52</span>, <span style="color:green;">+1.61</span>, <span style="color:green;">+2.22</span></td>
            <td style="text-align:center;" colspan="3"><span style="color:green;">+2.32</span>, <span style="color:green;">+2.30</span>, <span style="color:green;">+3.10</span></td>
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
