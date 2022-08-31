import sys, os
sys.path.append(os.getcwd())

import argparse
from pcflow.utils.data_utils.waymo_utils import creat_waymo_sf_data
from pcflow.datasets.kitti_sf_dataset import KittiSFdataset as KittiDataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create scene flow data')
    parser.add_argument('--dataset_type', type=str, default='waymo')
    args = parser.parse_args()
    if args.dataset_type == 'waymo':
        scene_id_list = list(range(100)) # can be any scene
        creat_waymo_sf_data(
            raw_data_path='./data/waymo',
            data_path='./data/waymo_sf',
            save_path='./data/waymo_sf', 
            scene_id_list=scene_id_list, 
            split='train',
            pc_num_features=8, # 8--waymo>=1.3, 6--waymo==1.2
            rm_ground=True,
            crop_params_save_path='./crop_params.npz',
            crop_params_load_path=None, # None means not using existing params
            num_workers=8)
    elif args.dataset_type == 'kitti':
        dataset = KittiDataset(npoints=16384, data_root='./data/kitti_tracking/training')
        dataset.create_data(save_path='./data/kitti_sf')