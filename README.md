<a name="FgWbR"></a>
# FH-Net: A Fast Hierarchical Network for Scene Flow Estimation on Real-world Point Clouds
![test](https://github.com/pigtigger/FH-Net/blob/main/demo/waymo.gif)
<a name="LEEHl"></a>
## Introduction
Estimating scene flow from real-world point clouds is a fundamental task for practical 3D vision. To alleviate the chronic shortage of real-world training data, we establish two new large-scale datasets, KITTI-SF and Waymo-SF, to this field by collecting lidar-scanned point clouds from public autonomous driving datasets and annotating the collected data through novel pseudo-labeling. On the real-world large-scale datasets, previous methods suffering high computational cost and latency. In this work, we propose a fast hierarchical network, FH-Net, which directly gets the key points flow through a lightweight Trans-flow layer utilizing the reliable local geometry prior, and optionally back-propagates the computed sparse flows through an inverse Trans-up layer to obtain hierarchical flows at different resolutions. In this way, our method achieves state-of-the-art performance and efficiency on the large-scale datasets.
## Requirements
- pytorch >= 1.7
- numpy
- numba
- [waymo-open-dataset-tf-2-6-0](https://github.com/waymo-research/waymo-open-dataset/blob/master/docs/quick_start.md)
- [mmcv](https://github.com/open-mmlab/mmcv)
- tqdm
<a name="CZOc8"></a>
## Usage
We construct two real-world scene flow datasets SF-Waymo and SF-KITTI, based on [Waymo 1.4](https://waymo.com/open/) and [KITTI](http://www.cvlibs.net/datasets/kitti/).

- For SF-Waymo,  download the Waymo raw data from [link_to_waymo_open_dataset](https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_4_0;tab=objects?pli=1&prefix=&forceOnObjectsSortingFiltering=false),  then put the training tfrecords (we use the first 100 scenes by default) into corresponding folders in `data/waymo/training`,  then run the following command to extract point clouds, 3D annotations, poses and other information form raw data.
```bash
python tools/waymo_extract.py
```
   After extracting data, the folder structure is the same as below :
   ```
   data
   ├── waymo
   │   ├── ImageSets
   │   └── training
   │       ├── segment-10017090168044687777_6380_000_6400_000_with_camera_labels.tfrecord
   │       └── ...
   └── waymo_sf
       ├── 000
       │   ├── calib
       │   ├── label_all
       │   ├── pose
       │   └── velodyne
       ├── 001
       └── ...
   ```
   Then create scene flow data by :
```bash
python tools/create_data.py --dataset_type waymo
```
The scene flow data will be saved to `data/waymo_sf/{scene_id}/`. A mini version of SF-Waymo is also provided [here](https://drive.google.com/drive/u/0/folders/1u9xeSnk_M2jVNwEDmr_Skr1teOpRqqxv) directly.

**Note:** We recommend using waymo 1.4, since we used segmentation labels to assist in the removal of the ground, and waymo 1.2 dataset has no segmentation labels. If you want to customize the data, you must ensure that waymo >= 1.3 and install waymo-open-dataset-tf-2-6-0. Another way is to retain the ground (more challenging), or use the ground removal parameters we provide in `crop_params.npz` (including the first 100 scenes), which you can load in `create_data.py`.

- For SF-KITTI,  download the KITTI raw data from [link_to_kitti_dataset](http://www.cvlibs.net/datasets/kitti/eval_tracking.php),  (note that only the point clouds and annotations are essential now, we will add more information to the dataset like color in the future),  then put the unzipped folder into `data/kitti/`,  and run : 
```bash
python tools/create_sf_data.py --dataset_type kitti
```
The processed SF-KITTI data is also provided [here](https://drive.google.com/drive/u/0/folders/1u9xeSnk_M2jVNwEDmr_Skr1teOpRqqxv) for download.

- We are integrating a complete scene flow code framework and plan to release it at 2022.10. The framework including data preparation, data augmentation, training, evluation, visualization and support most of the current methods. Our model code will be released at the same time, please wait patiently.

## Acknowledgments
This project is based on the following codebases.
- [mmdetection3d](https://github.com/open-mmlab/mmdetection3d)
- [Rigid3DSceneFlow](https://github.com/zgojcic/Rigid3DSceneFlow)
- [FlowNet3D](https://github.com/xingyul/flownet3d)

We thank the authors for releasing the code and provide support throughout the development of this project.


