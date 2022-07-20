<a name="FgWbR"></a>
# FH-Net: A Fast Hierarchical Network for Scene Flow Estimation on Real-world Point Clouds
![test](https://github.com/pigtigger/FH-Net/blob/main/demo/waymo.gif)
<a name="LEEHl"></a>
## Environment
- pytorch >= 1.7
- numpy
- [waymo-open-dataset](https://github.com/waymo-research/waymo-open-dataset/blob/master/docs/quick_start.md)
- [mmcv](https://github.com/open-mmlab/mmcv)
- tqdm
<a name="CZOc8"></a>
## Usage
We construct two real-world scene flow datasets SF-Waymo and SF-KITTI, based on [Waymo](https://waymo.com/open/) and [KITTI](http://www.cvlibs.net/datasets/kitti/).

- For SF-Waymo,  download the Waymo raw data from [link_to_waymo_open_dataset](https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_4_0;tab=objects?pli=1&prefix=&forceOnObjectsSortingFiltering=false),  then put the  tfrecords (we use the first 100 scenes by default) into corresponding folders in data/waymo/,  then run the following command to extract point clouds, 3D annotations, poses and other information form raw data.
```bash
python tools/extract_waymo_data.py
```
   After extracting data, the folder structure is the same as below :
   ```
   data
   └── waymo
       ├── 000
       │   ├── calib
       │   ├── label_all
       │   ├── pose
       │   └── velodyne
       └── 001
   ```
   Then create scene flow data by :
```bash
python tools/create_sf_data.py --dataset_type waymo
```

The scene flow data will be saved to `data/waymo_sf/{scene_id}/`. A mini version of SF-Waymo is also provided [here](https://drive.google.com/drive/u/0/folders/1u9xeSnk_M2jVNwEDmr_Skr1teOpRqqxv) directly.

- For SF-KITTI,  download the  KITTI raw data from [link_to_kitti_dataset](http://www.cvlibs.net/datasets/kitti/eval_tracking.php),  (note that only the point clouds and annotations are essential now, we will add more information to the dataset like color in the future),  then  put the unzipped folder into data/kitti/,  and  run : 
```bash
python tools/create_sf_data.py --dataset_type kitti
```
The processed SF-KITTI data is also provided [here](https://drive.google.com/drive/u/0/folders/1u9xeSnk_M2jVNwEDmr_Skr1teOpRqqxv) for download.

- We are still cleaning up the code of model. In addition, we plan to build a  whole framework for scene flow estimation including data preparation, data augmentation, training, evluation, visualization and support most of the current methods,  please wait patiently.

## Acknowledgments
This project is based on the following codebases.
- [mmdetection3d](https://github.com/open-mmlab/mmdetection3d)
- [Rigid3DSceneFlow](https://github.com/zgojcic/Rigid3DSceneFlow)
- [FlowNet3D](https://github.com/xingyul/flownet3d)

We thank the authors for releasing the code and provide support throughout the development of this project.


