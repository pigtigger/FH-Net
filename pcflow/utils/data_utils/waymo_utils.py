import numpy as np
import random
from collections import OrderedDict
from concurrent import futures as futures
from pathlib import Path
from skimage import io
from tqdm import tqdm

from torch.utils.data import Dataset
from utils.data_utils import box_np_ops
from utils.data_utils.crop_ground import CropGroundAuto


def get_kitti_info_path(idx,
                        prefix,
                        info_type,
                        file_tail,
                        relative_path,
                        exist_check):
    filename = '{:07d}'.format(idx) + file_tail
    scene_path = '{:03d}'.format(idx // 1000 % 1000)
    prefix = Path(prefix)
    file_path = Path(scene_path) / info_type / filename
    if exist_check and not (prefix / file_path).exists():
        raise ValueError('file not exist: {}'.format(file_path))
    if relative_path:
        return str(file_path)
    else:
        return str(prefix / file_path)


def get_image_path(idx,
                   prefix,
                   relative_path=True,
                   exist_check=True,
                   info_type='image_2'):
    return get_kitti_info_path(idx, prefix, info_type, '.png',
                               relative_path, exist_check)


def get_label_path(idx,
                   prefix,
                   relative_path=True,
                   exist_check=True,
                   info_type='label_2'):
    return get_kitti_info_path(idx, prefix, info_type, '.txt',
                               relative_path, exist_check)


def get_velodyne_path(idx,
                      prefix,
                      relative_path=True,
                      exist_check=True):
    return get_kitti_info_path(idx, prefix, 'velodyne', '.bin',
                               relative_path, exist_check)


def get_calib_path(idx,
                   prefix,
                   relative_path=True,
                   exist_check=True):
    return get_kitti_info_path(idx, prefix, 'calib', '.txt',
                               relative_path, exist_check)


def get_pose_path(idx,
                  prefix,
                  relative_path=True,
                  exist_check=True):
    return get_kitti_info_path(idx, prefix, 'pose', '.txt',
                               relative_path, exist_check)


def get_label_anno(label_path):
    annotations = {}
    annotations.update({
        'name': [],
        'truncated': [],
        'occluded': [],
        'alpha': [],
        'bbox': [],
        'dimensions': [],
        'location': [],
        'rotation_y': [],
        'track_id': []
    })
    with open(label_path, 'r') as f:
        lines = f.readlines()
    # if len(lines) == 0 or len(lines[0]) < 15:
    #     content = []
    # else:
    content = [line.strip().split(' ') for line in lines]
    num_objects = len([x[0] for x in content if x[0] != 'DontCare'])
    annotations['name'] = np.array([x[0] for x in content])
    num_gt = len(annotations['name'])
    annotations['truncated'] = np.array([float(x[1]) for x in content])
    annotations['occluded'] = np.array([int(x[2]) for x in content])
    annotations['alpha'] = np.array([float(x[3]) for x in content])
    annotations['bbox'] = np.array([[float(info) for info in x[4:8]]
                                    for x in content]).reshape(-1, 4)
    # dimensions will convert hwl format to standard lhw(camera) format.
    annotations['dimensions'] = np.array([[float(info) for info in x[8:11]]
                                          for x in content
                                          ]).reshape(-1, 3)[:, [2, 0, 1]]
    annotations['location'] = np.array([[float(info) for info in x[11:14]]
                                        for x in content]).reshape(-1, 3)
    annotations['rotation_y'] = np.array([float(x[14])
                                          for x in content]).reshape(-1)
    if len(content) != 0 and len(content[0]) == 16:  # have score
        annotations['score'] = np.array([float(x[15]) for x in content])
    else:
        annotations['score'] = np.zeros((annotations['bbox'].shape[0], ))
    annotations['track_id'] = [x[16] for x in content]
    index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
    annotations['index'] = np.array(index, dtype=np.int32)
    annotations['group_ids'] = np.arange(num_gt, dtype=np.int32)
    return annotations


def _extend_matrix(mat):
    mat = np.concatenate([mat, np.array([[0., 0., 0., 1.]])], axis=0)
    return mat


def get_waymo_frame_info(path,
                         label_info=True,
                         velodyne=True,
                         pc_num_features=6,
                         calib=True,
                         pose=True,
                         image=False,
                         with_imageshape=False,
                         extend_matrix=True,
                         frame_idx=[],
                         num_workers=8,
                         relative_path=True):
    """
    Get info of all frames:
    {
        [optional]points: [N, 3+] point cloud
        [optional, for kitti]image: {
            frame_idx: ...
            frame_path: ...
        }
        point_cloud: {
            num_features: 4
            velodyne_path: ...
        }
        [optional, for kitti]calib: {
            R0_rect: ...
            Tr_velo_to_cam: ...
            P2: ...
        }
        annos: {
            location: [num_gt, 3] array
            dimensions: [num_gt, 3] array
            rotation_y: [num_gt] angle array
            name: [num_gt] ground truth name array
            [optional]difficulty: kitti difficulty
            [optional]group_ids: used for multi-part object
        }
    }
    """
    root_path = Path(path)

    def map_func(idx):
        info = {'idx': idx}
        pc_info = {'num_features': pc_num_features}
        calib_info = {}

        annotations = None
        if velodyne:
            pc_info['velodyne_path'] = get_velodyne_path(
                idx, path, relative_path)
            points = np.fromfile(
                Path(path) / pc_info['velodyne_path'], dtype=np.float32)
            points = np.copy(points).reshape(-1, pc_info['num_features'])
        info['point_cloud'] = pc_info
        if label_info:
            label_path = get_label_path(
                idx,
                path,
                relative_path,
                info_type='label_all')
            if relative_path:
                label_path = str(root_path / label_path)
            annotations = get_label_anno(label_path)
        if image:
            image_info = {}
            image_info['image_path'] = get_image_path(
                idx,
                path,
                relative_path,
                info_type='image_0')
            if with_imageshape:
                img_path = image_info['image_path']
                if relative_path:
                    img_path = str(root_path / img_path)
                image_info['image_shape'] = np.array(
                    io.imread(img_path).shape[:2], dtype=np.int32)
            info['image'] = image_info
        if calib:
            calib_path = get_calib_path(
                idx, path, relative_path=False)
            with open(calib_path, 'r') as f:
                lines = f.readlines()
            P0 = np.array([float(info) for info in lines[0].split(' ')[1:13]
                           ]).reshape([3, 4])
            P1 = np.array([float(info) for info in lines[1].split(' ')[1:13]
                           ]).reshape([3, 4])
            P2 = np.array([float(info) for info in lines[2].split(' ')[1:13]
                           ]).reshape([3, 4])
            P3 = np.array([float(info) for info in lines[3].split(' ')[1:13]
                           ]).reshape([3, 4])
            P4 = np.array([float(info) for info in lines[4].split(' ')[1:13]
                           ]).reshape([3, 4])
            if extend_matrix:
                P0 = _extend_matrix(P0)
                P1 = _extend_matrix(P1)
                P2 = _extend_matrix(P2)
                P3 = _extend_matrix(P3)
                P4 = _extend_matrix(P4)
            R0_rect = np.array([
                float(info) for info in lines[5].split(' ')[1:10]
            ]).reshape([3, 3])
            if extend_matrix:
                rect_4x4 = np.zeros([4, 4], dtype=R0_rect.dtype)
                rect_4x4[3, 3] = 1.
                rect_4x4[:3, :3] = R0_rect
            else:
                rect_4x4 = R0_rect

            Tr_velo_to_cam = np.array([
                float(info) for info in lines[6].split(' ')[1:13]
            ]).reshape([3, 4])
            if extend_matrix:
                Tr_velo_to_cam = _extend_matrix(Tr_velo_to_cam)
            calib_info['P0'] = P0
            calib_info['P1'] = P1
            calib_info['P2'] = P2
            calib_info['P3'] = P3
            calib_info['P4'] = P4
            calib_info['R0_rect'] = rect_4x4
            calib_info['Tr_velo_to_cam'] = Tr_velo_to_cam
            info['calib'] = calib_info
        if pose:
            pose_path = get_pose_path(
                idx, path, relative_path=False)
            info['pose'] = np.loadtxt(pose_path)

        if annotations is not None:
            info['annos'] = annotations
            info['annos']['camera_id'] = info['annos'].pop('score')
            add_difficulty_to_annos(info)
        return info

    print('Start to load infos.')
    with futures.ThreadPoolExecutor(num_workers) as executor:
        frame_infos = list(tqdm(executor.map(map_func, frame_idx), total=len(frame_idx), ncols=100))
    
    frame_infos_dict = {}
    for info in frame_infos:
        frame_infos_dict[info['idx']] = info
    return frame_infos_dict


def kitti_anno_to_label_file(annos, folder):
    folder = Path(folder)
    for anno in annos:
        image_idx = anno['metadata']['image_idx']
        label_lines = []
        for j in range(anno['bbox'].shape[0]):
            label_dict = {
                'name': anno['name'][j],
                'alpha': anno['alpha'][j],
                'bbox': anno['bbox'][j],
                'location': anno['location'][j],
                'dimensions': anno['dimensions'][j],
                'rotation_y': anno['rotation_y'][j],
                'score': anno['score'][j],
            }
            label_line = kitti_result_line(label_dict)
            label_lines.append(label_line)
        label_file = folder / '{:07d}.txt'.format(image_idx)
        label_str = '\n'.join(label_lines)
        with open(label_file, 'w') as f:
            f.write(label_str)


def add_difficulty_to_annos(info):
    min_height = [40, 25,
                  25]  # minimum height for evaluated groundtruth/detections
    max_occlusion = [
        0, 1, 2
    ]  # maximum occlusion level of the groundtruth used for evaluation
    max_trunc = [
        0.15, 0.3, 0.5
    ]  # maximum truncation level of the groundtruth used for evaluation
    annos = info['annos']
    dims = annos['dimensions']  # lhw format
    bbox = annos['bbox']
    height = bbox[:, 3] - bbox[:, 1]
    occlusion = annos['occluded']
    truncation = annos['truncated']
    diff = []
    easy_mask = np.ones((len(dims), ), dtype=np.bool)
    moderate_mask = np.ones((len(dims), ), dtype=np.bool)
    hard_mask = np.ones((len(dims), ), dtype=np.bool)
    i = 0
    for h, o, t in zip(height, occlusion, truncation):
        if o > max_occlusion[0] or h <= min_height[0] or t > max_trunc[0]:
            easy_mask[i] = False
        if o > max_occlusion[1] or h <= min_height[1] or t > max_trunc[1]:
            moderate_mask[i] = False
        if o > max_occlusion[2] or h <= min_height[2] or t > max_trunc[2]:
            hard_mask[i] = False
        i += 1
    is_easy = easy_mask
    is_moderate = np.logical_xor(easy_mask, moderate_mask)
    is_hard = np.logical_xor(hard_mask, moderate_mask)

    for i in range(len(dims)):
        if is_easy[i]:
            diff.append(0)
        elif is_moderate[i]:
            diff.append(1)
        elif is_hard[i]:
            diff.append(2)
        else:
            diff.append(-1)
    annos['difficulty'] = np.array(diff, np.int32)
    return diff


def kitti_result_line(result_dict, precision=4):
    prec_float = '{' + ':.{}f'.format(precision) + '}'
    res_line = []
    all_field_default = OrderedDict([
        ('name', None),
        ('truncated', -1),
        ('occluded', -1),
        ('alpha', -10),
        ('bbox', None),
        ('dimensions', [-1, -1, -1]),
        ('location', [-1000, -1000, -1000]),
        ('rotation_y', -10),
        ('score', 0.0),
    ])
    res_dict = [(key, None) for key, val in all_field_default.items()]
    res_dict = OrderedDict(res_dict)
    for key, val in result_dict.items():
        if all_field_default[key] is None and val is None:
            raise ValueError('you must specify a value for {}'.format(key))
        res_dict[key] = val

    for key, val in res_dict.items():
        if key == 'name':
            res_line.append(val)
        elif key in ['truncated', 'alpha', 'rotation_y', 'score']:
            if val is None:
                res_line.append(str(all_field_default[key]))
            else:
                res_line.append(prec_float.format(val))
        elif key == 'occluded':
            if val is None:
                res_line.append(str(all_field_default[key]))
            else:
                res_line.append('{}'.format(val))
        elif key in ['bbox', 'dimensions', 'location']:
            if val is None:
                res_line += [str(v) for v in all_field_default[key]]
            else:
                res_line += [prec_float.format(v) for v in val]
        else:
            raise ValueError('unknown key. supported key:{}'.format(
                res_dict.keys()))
    return ' '.join(res_line)


def creat_waymo_sf_data(raw_data_path=None,
                        data_path=None, 
                        save_path=None, 
                        scene_id_list=[], 
                        split='train',
                        pc_num_features=8, # 8 -- waymo > 1.3, 6 -- waymo 1.2
                        rm_ground=True,
                        crop_params_save_path=None,
                        crop_params_load_path=None,
                        num_workers=8):
    """Create info file of waymo dataset. Given the data created by mmdetection3d, 
    generate scene flow data.

    Args:
        data_path (str): Path of the data root.
        save_path (str | None): Path to save the info file.
        scene_id_list (List[int]): List of the scene id to create scene flow.
        split (str, optional): Train, val or other split of the dataset.
        relative_path (bool, optional)
        pc_num_features (int): 8 or 6 for waymo
        num_workers (int)
    """
    if data_path is None:
        data_path = raw_data_path
    if save_path is None:
        save_path = Path(data_path)
    else:
        save_path = Path(save_path)
    frame_idx = get_frame_idx(raw_data_path, scene_id_list, split=split)
    frame_infos_dict = get_waymo_frame_info(
        data_path,
        label_info=True,
        velodyne=True,
        pc_num_features=pc_num_features,
        calib=True,
        pose=True,
        image=False,
        with_imageshape=False,
        extend_matrix=True,
        frame_idx=frame_idx,
        num_workers=num_workers,
        relative_path=True)
    if rm_ground:
        croper = CropGroundAuto()
        crop_pointer_dict = generate_crop_params(
            croper=croper,
            frame_infos_dict=frame_infos_dict, 
            scene_id_list=scene_id_list, 
            frame_idx=frame_idx, 
            data_path=data_path, 
            split=split, 
            pc_num_features=pc_num_features,
            save_path=crop_params_save_path,
            load_path=crop_params_load_path)
    else:
        croper, crop_pointer_dict = None, None
    generate_scene_flow(
        data_path,
        save_path,
        frame_idx,
        frame_infos_dict,
        remove_outside=False,
        pc_num_features=pc_num_features,
        croper=croper,
        crop_pointer_dict=crop_pointer_dict,
        downsample=None,
        num_workers=num_workers)
    print('Finish.')

def generate_crop_params(croper,
                         frame_infos_dict, 
                         scene_id_list, 
                         frame_idx, 
                         data_path, 
                         split, 
                         pc_num_features,
                         save_path=None,
                         load_path=None):
    """Generate crop parameters for remove ground.
    """
    if load_path is not None:
        crop_pointer_dict = croper.load_crop_pointer_dict(load_path)
        print('Load {} sceces parameters of croping ground from '.format(
            len(crop_pointer_dict.keys())))
        cur_s_id_list = [_ for _ in scene_id_list if _ not in crop_pointer_dict.keys()]
    else:
        crop_pointer_dict = {}
        cur_s_id_list = scene_id_list
    if cur_s_id_list != []:
        assert pc_num_features == 8, 'Need segmentation label.'
        print('Start to compute parameters of croping ground.')
        f_idx_array = np.array(frame_idx)
        s_idx_array = (f_idx_array // 1000).astype(np.int) % 1000
        perfix = 0 if split.lower() == 'train' else 1
        try:
            for s_id in tqdm(cur_s_id_list, total=len(cur_s_id_list), ncols=100):
                s_f_id = f_idx_array[s_idx_array == s_id]
                # f_num = int(s_f_id.max()) + 1
                f_num = len(s_f_id)
                # assert f_num == len(s_f_id)

                crop_pointer_scene = croper.gen_crop_pointer_scene(f_num)
                croper.reset_eval_thresh()
                label_ind = np.zeros(f_num, dtype=np.bool)
                record_pointer = crop_pointer_scene[24]
                for f_id in range(f_num):
                    f_id_ = int('{}{:03d}{:03d}'.format(perfix, s_id, f_id))
                    pc_path = str(Path(data_path) / frame_infos_dict[f_id_]['point_cloud']['velodyne_path'])
                    pc = np.fromfile(pc_path, dtype=np.float32).reshape(-1,8)
                    seg_label = pc[:, 7].astype(np.int32)
                    if seg_label.any():
                        pc_lidar = pc[:, :3]
                        record_pointer = croper.auto_crop_one_frame(pc_lidar, seg_label, record_pointer)
                        _min = max(f_id - 2, 0)
                        _max = f_id + 5
                        crop_pointer_scene[_min:_max] = record_pointer
                        label_ind[_min:_max] = True

                flag = True
                for f_id in range(f_num):
                    if flag and label_ind[f_id]:
                        crop_pointer_scene[:f_id] = crop_pointer_scene[f_id]
                        flag = False
                    elif (not flag) and (~label_ind[f_id]):
                        crop_pointer_scene[f_id] = crop_pointer_scene[f_id-1]
                    if (f_id == (f_num-1)) and flag:
                        print(f'Warning: No segmentation label in scene {s_id}.')                
                for f_id in range(f_num):
                    if label_ind[f_num-f_id-1]:
                        crop_pointer_scene[:f_id] = crop_pointer_scene[f_id]
                        break
                crop_pointer_dict[s_id] = crop_pointer_scene
        except Exception as err:
            if crop_pointer_dict != {}:
                croper.save_crop_pointer_dict(save_path, crop_pointer_dict)
                print('An error has occurred. Now saving {} scenes parameters of croping ground to '.format(
                    len(crop_pointer_dict.keys())) + save_path)
            raise err
        if save_path is not None:
            croper.save_crop_pointer_dict(save_path, crop_pointer_dict)
            print('Save parameters of croping ground to ' + save_path)
    return crop_pointer_dict

def generate_scene_flow(data_path,
                        save_path,
                        frame_idx,
                        frame_infos_dict,
                        remove_outside=False,
                        pc_num_features=8,
                        croper=None,
                        crop_pointer_dict=None,
                        downsample=None,
                        num_workers=8):
    """Create scene flow dataset.
    """
    def map_func(idx):
        err_count = 0
        err_tole = 100
        try:
            if idx + 1 in frame_infos_dict:
                s_id = idx // 1000 % 1000
                f_id = idx % 1000

                info1 = frame_infos_dict[idx]
                info2 = frame_infos_dict[idx+1]

                pc_info1 = info1['point_cloud']
                pc_info2 = info2['point_cloud']
                pc_path1 = str(Path(data_path) / pc_info1['velodyne_path'])
                pc_path2 = str(Path(data_path) / pc_info2['velodyne_path'])
                pc1 = np.fromfile(
                    pc_path1, dtype=np.float32, count=-1).reshape([-1, pc_num_features])
                pc2 = np.fromfile(
                    pc_path2, dtype=np.float32, count=-1).reshape([-1, pc_num_features])

                calib1 = info1['calib']
                calib2 = info2['calib']
                rect1 = calib1['R0_rect']
                rect2 = calib2['R0_rect']
                Trv2c1 = calib1['Tr_velo_to_cam']
                Trv2c2 = calib2['Tr_velo_to_cam']
                P2_1 = calib1['P2']
                P2_2 = calib2['P2']

                if remove_outside:
                    image_info1 = info1['image']
                    pc1 = box_np_ops.remove_outside_points(
                        pc1, rect1, Trv2c1, P2_1, image_info1['image_shape'])
                    image_info2 = info2['image']
                    pc2 = box_np_ops.remove_outside_points(
                        pc2, rect2, Trv2c2, P2_2, image_info2['image_shape'])

                # points_v = points_v[points_v[:, 0] > 0]
                annos1 = info1['annos']
                annos2 = info2['annos']
                num_obj1 = len([n for n in annos1['name'] if n != 'DontCare'])
                num_obj2 = len([n for n in annos2['name'] if n != 'DontCare'])
                # annos1 = filter_kitti_anno(annos1, ['DontCare'])
                # annos2 = filter_kitti_anno(annos2, ['DontCare'])
                dims1 = annos1['dimensions'][:num_obj1]
                dims2 = annos2['dimensions'][:num_obj2]
                loc1 = annos1['location'][:num_obj1]
                loc2 = annos2['location'][:num_obj2]
                rots1 = annos1['rotation_y'][:num_obj1]
                rots2 = annos2['rotation_y'][:num_obj2]
                gt_boxes_camera1 = np.concatenate([loc1, dims1, rots1[..., np.newaxis]],
                                                axis=1)
                gt_boxes_camera2 = np.concatenate([loc2, dims2, rots2[..., np.newaxis]],
                                                axis=1)
                gt_boxes_lidar1 = box_np_ops.box_camera_to_lidar(
                    gt_boxes_camera1, rect1, Trv2c1)
                gt_boxes_lidar2 = box_np_ops.box_camera_to_lidar(
                    gt_boxes_camera2, rect2, Trv2c2)
                T_boxes2lidar1 = box_np_ops.get_T_box_to_lidar(gt_boxes_lidar1)
                T_boxes2lidar2 = box_np_ops.get_T_box_to_lidar(gt_boxes_lidar2)
                T_lidar2boxes1 = np.linalg.inv(T_boxes2lidar1)
                pc1_xyz = pc1[:, :3]
                pc2_xyz = pc2[:, :3]
                pc1_pad = np.concatenate([pc1_xyz,np.ones((pc1.shape[0],1))], axis=-1)
                fg_ind1 = box_np_ops.points_in_rbbox(pc1_xyz, gt_boxes_lidar1)
                fg_ind2 = box_np_ops.points_in_rbbox(pc2_xyz, gt_boxes_lidar2)
                bg_ind1 = ~(fg_ind1.sum(axis=1).astype(np.bool))
                bg_ind2 = ~(fg_ind2.sum(axis=1).astype(np.bool))
                bg_pc1_xyz = pc1_xyz[bg_ind1]
                bg_pc2_xyz = pc2_xyz[bg_ind2]
                
                if croper is not None:
                    crop_pointer1 = crop_pointer_dict[s_id][f_id]
                    crop_pointer2 = crop_pointer_dict[s_id][f_id+1]
                    crop_mask1 = croper.compute_crop(bg_pc1_xyz[:,:3], crop_pointer1)
                    crop_mask2 = croper.compute_crop(bg_pc2_xyz[:,:3], crop_pointer2)
                    bg_pc1 = pc1[bg_ind1][crop_mask1]
                    bg_pc2 = pc2[bg_ind2][crop_mask2]
                    bg_pc1_pad = pc1_pad[bg_ind1][crop_mask1]
                else:
                    bg_pc1 = pc1[bg_ind1]
                    bg_pc2 = pc2[bg_ind2]
                    bg_pc1_pad = pc1_pad[bg_ind1]
                
                # compute background flow
                pose1 = info1['pose']
                pose2 = info2['pose']
                bg_flow = (bg_pc1_pad @ pose1.T @ np.linalg.inv(pose2).T - bg_pc1_pad)[:,:3]

                # compute foreground flow
                track_id1 = annos1['track_id'][:num_obj1]
                track_id2 = annos2['track_id'][:num_obj2]
                fg_pc1, fg_pc2, fg_flow = [], [], []
                for i in range(len(track_id1)):
                    if track_id1[i] in track_id2:
                        i2 = track_id2.index(track_id1[i])
                        fg_ind1_ = fg_ind1[:, i]
                        fg_ind2_ = fg_ind2[:, i2]
                        fg_pc1.append(pc1[fg_ind1_])
                        fg_pc2.append(pc2[fg_ind2_])
                        pc1_obj = pc1_pad[fg_ind1_]
                        T_lidar2boxes1_obj = T_lidar2boxes1[i]
                        T_boxes2lidar2_obj = T_boxes2lidar2[i2]
                        pc1_obj_in_2 = pc1_obj @ T_lidar2boxes1_obj.T @ T_boxes2lidar2_obj.T
                        flow_obj = (pc1_obj_in_2 - pc1_obj)[:,:3]
                        fg_flow.append(flow_obj)
                fg_pc1 = np.concatenate(fg_pc1, axis=0) if fg_pc1 != [] else np.empty((0,pc1.shape[-1]))
                fg_pc2 = np.concatenate(fg_pc2, axis=0) if fg_pc2 != [] else np.empty((0,pc2.shape[-1]))
                fg_flow = np.concatenate(fg_flow, axis=0) if fg_flow != [] else np.empty((0,3))
                
                new_pc1 = np.concatenate([bg_pc1, fg_pc1], axis=0)
                new_pc2 = np.concatenate([bg_pc2, fg_pc2], axis=0)
                gt_flow = np.concatenate([bg_flow, fg_flow], axis=0).astype(np.float32)
                fg_indices1 = np.concatenate([
                    np.zeros(bg_pc1.shape[0], dtype=np.bool),
                    np.ones(fg_pc1.shape[0], dtype=np.bool)], axis=0)
                fg_indices2 = np.concatenate([
                    np.zeros(bg_pc2.shape[0], dtype=np.bool),
                    np.ones(fg_pc2.shape[0], dtype=np.bool)], axis=0)
                
                if (downsample is not None) and (isinstance(downsample, int)):
                    n1 = new_pc1.shape[0]
                    n2 = new_pc2.shape[0]
                    if downsample < 10000:
                        samp1 = random.sample(range(n1), downsample)
                        samp2 = random.sample(range(n2), downsample)
                    else:
                        samp1 = np.random.choice(n1, size=downsample, replace=False)
                        samp2 = np.random.choice(n2, size=downsample, replace=False)
                    new_pc1 = new_pc1[samp1]
                    new_pc2 = new_pc2[samp2]
                    gt_flow = gt_flow[samp1]
                    fg_indices1 = fg_indices1[samp1]
                    fg_indices2 = fg_indices2[samp2]
                
                save_dir = Path(save_path) / '{:03d}'.format(s_id) / 'sf_data'
                save_dir.mkdir(parents=True, exist_ok=True)
                file_save_path = save_dir / '{:07d}'.format(idx)
                np.savez(file_save_path, pc1=new_pc1, pc2=new_pc2, gt=gt_flow, fg_index=fg_indices1, fg_index_t=fg_indices2)
        except Exception as err:
            print('An error occurred during the generation of {:07d}.npz'.format(idx))
            err_count += 1
            if err_count <= err_tole:
                print(err)
            else:
                raise err
        return None
    
    print('Start to generate flow data.')
    with futures.ThreadPoolExecutor(num_workers) as executor:
        list(tqdm(executor.map(map_func, frame_idx), total=len(frame_idx), ncols=100))

def get_frame_idx(self, data_path, scene_id_list, split):
    imageset_folder = Path(data_path) / 'ImageSets'
    imageset_path = str(imageset_folder / (split.lower() + '.txt'))
    with open(imageset_path, 'r') as f:
        lines = f.readlines()
    frame_idx = []
    for line in lines:
        scene_id = int(line[1:4])
        if scene_id in scene_id_list:
            frame_idx.append(int(line))
    return frame_idx