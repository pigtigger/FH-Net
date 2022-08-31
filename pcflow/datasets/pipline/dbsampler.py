import copy
from pathlib import Path
import math
import random
import numpy as np
import torch
import mmcv
import os

from pcflow.utils.data_utils.structures import BaseSceneFlow
from pcflow.utils.data_utils.box_np_ops import rotation_points_single_angle


class BatchSampler:
    """Class for sampling specific category of ground truths.

    Args:
        sample_list (list[dict]): List of samples.
        name (str | None): The category of samples. Default: None.
        epoch (int | None): Sampling epoch. Default: None.
        shuffle (bool): Whether to shuffle indices. Default: False.
        drop_reminder (bool): Drop reminder. Default: False.
    """

    def __init__(self,
                 sampled_list,
                 name=None,
                 epoch=None,
                 shuffle=True,
                 drop_reminder=False):
        self._sampled_list = sampled_list
        self._indices = np.arange(len(sampled_list))
        if shuffle:
            np.random.shuffle(self._indices)
        self._idx = 0
        self._example_num = len(sampled_list)
        self._name = name
        self._shuffle = shuffle
        self._epoch = epoch
        self._epoch_counter = 0
        self._drop_reminder = drop_reminder

    def _sample(self, num):
        """Sample specific number of ground truths and return indices.

        Args:
            num (int): Sampled number.

        Returns:
            list[int]: Indices of sampled ground truths.
        """
        if self._idx + num >= self._example_num:
            # ret = self._indices[self._idx:].copy()
            ret = self._indices[-num:].copy()
            self._reset()
        else:
            ret = self._indices[self._idx:self._idx + num]
            self._idx += num
        return ret

    def _reset(self):
        """Reset the index of batchsampler to zero."""
        assert self._name is not None
        # print("reset", self._name)
        if self._shuffle:
            np.random.shuffle(self._indices)
        self._idx = 0

    def sample(self, num):
        """Sample specific number of ground truths.

        Args:
            num (int): Sampled number.

        Returns:
            list[dict]: Sampled ground truths.
        """
        indices = self._sample(num)
        return [self._sampled_list[i] for i in indices]


class DataBaseSampler(object):
    """Class for sampling data from the ground truth database.

    Args:
        info_path (str): Path of groundtruth database info.
        data_root (str): Path of groundtruth database.
        rate (float): Rate of actual sampled over maximum sampled number.
        prepare (dict): Name of preparation functions and the input value.
        sample_groups (dict): Sampled classes and numbers.
        classes (list[str]): List of classes. Default: None.
        filter_by_min_points (dict): Different number of minimum points
            needed for different categories of ground truths.
        point_cloud_range (list[float]): Point cloud range.
        pillar_size (float): Pillar size when calculating available zone.
        pillar_points_thresh (int): Min points in a pillar when calculating available zone.
        height_axis (int): Vertical axis.
        db_load_dim (int): Load dim of database files.
        db_use_dim (int): Use dim of database files.
        sim_velocity (dict): Simulated velocity of sampled object.
    """

    def __init__(self,
                 info_path,
                 data_root,
                 rate,
                 prepare,
                 sample_groups,
                 classes=None,
                 filter_by_min_points=None,
                 point_cloud_range=[-75.2, -75.2, -2., 75.2, 75.2, 4.],
                 height_axis=2,
                 pillar_size=1.6,
                 pillar_points_thresh=10,
                 db_load_dim=5,
                 db_use_dim=[0,1,2,3,4],
                 sim_velocity={
                    'Car':60/3.6/5,
                    'Pedestrian':5/3.6/5,
                    'Cyclist':30/3.6/5}):
        super().__init__()
        self.data_root = Path(data_root) if data_root else None
        self.info_path = info_path
        self.rate = rate
        self.prepare = prepare
        self.classes = classes
        self.cat2label = {name: i for i, name in enumerate(classes)}
        self.label2cat = {i: name for i, name in enumerate(classes)}
        # self.points_loader = mmcv.build_from_cfg(points_loader, PIPELINES)

        db_infos = mmcv.load(info_path)
        if isinstance(filter_by_min_points, dict):
            db_infos = self.filter_by_min_points(db_infos, filter_by_min_points)

        # filter database infos
        # from mmdet3d.utils import get_root_logger
        # logger = get_root_logger()
        # for k, v in db_infos.items():
        #     logger.info(f'load {len(v)} {k} database infos')
        # for prep_func, val in prepare.items():
        #     db_infos = getattr(self, prep_func)(db_infos, val)
        # logger.info('After filter database:')
        # for k, v in db_infos.items():
        #     logger.info(f'load {len(v)} {k} database infos')

        self.db_infos = db_infos

        # load sample groups
        # TODO: more elegant way to load sample groups
        self.sample_groups = []
        for name, num in sample_groups.items():
            self.sample_groups.append({name: int(num)})

        self.group_db_infos = self.db_infos  # just use db_infos
        self.sample_classes = []
        self.sample_max_nums = []
        for group_info in self.sample_groups:
            self.sample_classes += list(group_info.keys())
            self.sample_max_nums += list(group_info.values())

        self.sampler_dict = {}
        for k, v in self.group_db_infos.items():
            self.sampler_dict[k] = BatchSampler(v, k, shuffle=True)
        # TODO: No group_sampling currently

        self.bev_axis = [0,1,2]
        self.height_axis = height_axis
        self.bev_axis.remove(height_axis)
        point_cloud_range = torch.tensor(point_cloud_range, dtype=torch.float32)
        self.min_bound = point_cloud_range[self.bev_axis]
        self.max_bound = point_cloud_range[[self.bev_axis[0]+3, self.bev_axis[1]+3]]
        self.pillar_size = pillar_size
        self.grid_size = (self.max_bound - self.min_bound + 0.001) // pillar_size
        self.pillar_points_thresh = pillar_points_thresh
        self.db_load_dim = db_load_dim
        if isinstance(db_use_dim, int):
            self.db_use_dim = list(range(db_use_dim))
        else:
            self.db_use_dim = db_use_dim
        self.sim_velocity = sim_velocity
    
    def __call__(self, data_dict):
        """Call function to dbsampling.

        Args:
            data_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after dbsampling, 'sf_data' is updated in \
                the result dict.
        """
        sf_data = data_dict['sf_data']
        num_labels = data_dict['num_labels_frame']
        data_dict['sf_data'] = self.sample_all(sf_data, num_labels)
        return data_dict

    
    @staticmethod
    def filter_by_difficulty(db_infos, removed_difficulty):
        """Filter ground truths by difficulties.

        Args:
            db_infos (dict): Info of groundtruth database.
            removed_difficulty (list): Difficulties that are not qualified.

        Returns:
            dict: Info of database after filtering.
        """
        new_db_infos = {}
        for key, dinfos in db_infos.items():
            new_db_infos[key] = [
                info for info in dinfos
                if info['difficulty'] not in removed_difficulty
            ]
        return new_db_infos

    @staticmethod
    def filter_by_min_points(db_infos, min_gt_points_dict):
        """Filter ground truths by number of points in the bbox.

        Args:
            db_infos (dict): Info of groundtruth database.
            min_gt_points_dict (dict): Different number of minimum points
                needed for different categories of ground truths.

        Returns:
            dict: Info of database after filtering.
        """
        for name, min_num in min_gt_points_dict.items():
            min_num = int(min_num)
            if min_num > 0:
                filtered_infos = []
                for info in db_infos[name]:
                    if info['num_points_in_gt'] >= min_num:
                        filtered_infos.append(info)
                db_infos[name] = filtered_infos
        return db_infos
    
    def sample_all(self, sf_data: BaseSceneFlow, num_labels: list) -> BaseSceneFlow:
        """Sampling all categories of bboxes.

        Args:
            sf_data (:obj:`BaseSceneFlow`): Scene flow data.
            num_labels (list): Number of labels of each category.

        """
        sampled_num_dict = {}
        sample_num_per_class = []
        for class_name, max_sample_num in zip(self.sample_classes,
                                              self.sample_max_nums):
            class_label = self.cat2label[class_name]
            sampled_num = int(max_sample_num - num_labels[class_label])
            sampled_num = np.round(self.rate * sampled_num).astype(np.int64)
            sampled_num = np.max(sampled_num, 0)
            sampled_num_dict[class_name] = sampled_num
            sample_num_per_class.append(sampled_num)

        sampled = []
        sampled_gt_bboxes = []
        avail_zone = self.get_avail_zone(sf_data.pc1[:, self.bev_axis])
        avail_zone_size = len(avail_zone)
        sample_num_total = sum(sample_num_per_class)
        if avail_zone_size >= sample_num_total:
            avail_zone = avail_zone[torch.randint(avail_zone_size, (sample_num_total,))]
        else:
            avail_zone_extra = torch.stack([
                torch.randint(self.grid_size[0], (sample_num_total-avail_zone_size,)),
                torch.randint(self.grid_size[1], (sample_num_total-avail_zone_size,))],
                dim=1)
            avail_zone = torch.cat([avail_zone, avail_zone_extra], dim=0)
        paste_position = avail_zone * (self.pillar_size + torch.rand(
            (sample_num_total, 2), dtype=torch.float32))
        paste_position = torch.cat(
            [paste_position, torch.zeros((len(paste_position), 1))], dim=1)
        

        for class_name, sampled_num in zip(self.sample_classes,
                                           sample_num_per_class):
            if sampled_num > 0:
                sampled_cls = self.sampler_dict[class_name].sample(sampled_num)
                sampled_cls = copy.deepcopy(sampled_cls)
                sampled += sampled_cls
        
        if len(sampled) > 0:
            s_sf_list = []
            count = 0
            for info in sampled:
                file_path = self.data_root / info['path'] if self.data_root else info['path']
                s_pc = np.fromfile(file_path, dtype=np.float32).reshape(-1, self.db_load_dim)
                s_pc = s_pc[:, self.db_use_dim]
                s_pc1, s_pc2, s_sf = self.generate_sampled_sf(
                    s_pc, paste_position[count].numpy(), info['box3d_lidar'][6], info['name'])
                if sf_data.with_mask:
                    fg_mask1 = np.ones((len(s_pc1),), dtype=np.bool)
                    fg_mask2 = np.ones((len(s_pc2),), dtype=np.bool)
                    s_sf_list.append(BaseSceneFlow(s_pc1, s_pc2, s_sf, fg_mask1, fg_mask2))
                else:
                    s_sf_list.append(BaseSceneFlow(s_pc1, s_pc2, s_sf))
                count += 1
            sf_data = BaseSceneFlow.cat([sf_data] + s_sf_list)
        
        return sf_data
        
    def get_avail_zone(self, pc_bev: torch.Tensor):
        pc_pillar = pc_bev // self.pillar_size
        mask_in = pc_pillar[:, 0] >= 0 \
            & pc_pillar[:, 0] < self.grid_size \
            & pc_pillar[:, 1] >= 0 \
            & pc_pillar[:, 1] < self.grid_size
        pc_pillar = pc_pillar[mask_in]
        avoid_pillar, count = torch.unique(pc_pillar, return_counts=True, dim=0)
        avoid_pillar = avoid_pillar[count > self.pillar_points_thresh]
        avoid_pillar = avoid_pillar.transpose(1, 0)
        avail_map = torch.ones((self.grid_size[0], self.grid_size[1]), dtype=torch.bool)
        avail_map[avoid_pillar[0], avoid_pillar[1]] = False
        avail_zone = torch.nonzero(avail_map)
        return avail_zone
    
    def generate_sampled_sf(self, pc, trans1, theta1, cls_name):
        num_points = len(pc)
        pc1_idx = random.sample(
            range(num_points), math.floor(num_points*0.8))
        pc2_idx = random.sample(
            range(num_points), math.floor(num_points*0.8))
        pc1_ori = pc[pc1_idx]
        pc2_ori = pc[pc2_idx]

        sim_velo = self.sim_velocity[cls_name]
        sim_velo = np.random.rand() * sim_velo
        # TODO: Hard, height_axis=2
        shift_x = -sim_velo * np.sin(theta1)
        shift_y = -sim_velo * np.cos(theta1)
        shift_obj = np.array([shift_x, shift_y, 0], dtype=np.float32)
        shift_self = np.random.randn() * 0.56 + 1.67 # 30+-10 km/h
        shift_self = np.array([-shift_self, 0, 0], dtype=np.float32)
        trans2 = trans1 + shift_obj + shift_self

        theta2_noise = np.random.randn() * 0.17 # +-10 degree

        pc1 = pc1_ori + trans1
        pc2 = pc2_ori + trans2
        pc1_pos2, _ = rotation_points_single_angle(pc1_ori, theta2_noise, axis=self.height_axis)
        pc1_pos2 += trans2
        sf = pc1_pos2 - pc1
        return pc1, pc2, sf
        
    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f' data_root={self.data_root},'
        repr_str += f' info_path={self.info_path},'
        repr_str += f' rate={self.rate},'
        repr_str += f' prepare={self.prepare},'
        repr_str += f' classes={self.classes},'
        repr_str += f' sample_groups={self.sample_groups}'
        repr_str += f' sim_velocity={self.sim_velocity}'
        return repr_str