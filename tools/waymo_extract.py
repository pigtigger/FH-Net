import os
from .waymo_converter import Waymo2KITTI
from tqdm import tqdm

load_dir = './data/waymo/waymo_format/training' # raw data
save_dir = './data/waymo_sf'
scene_num = len(os.listdir(load_dir))

if __name__ == '__main__':
    waymo = Waymo2KITTI(load_dir, save_dir, str(0), scene_num=scene_num)
    print('Total scene :', scene_num)
    print('Strat convert waymo please wait...')
    for i in tqdm(range(scene_num), total=scene_num, ncols=100):
        waymo.convert_one(i)
    print('DONE!')