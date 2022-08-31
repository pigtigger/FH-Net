import os
import glob
import numpy as np
from numpy.linalg import inv
import pcflow.utils.data_utils.kitti_utils as kitti_utils
import pcflow.utils.data_utils.kitti_oxts as kitti_oxts
from tqdm import tqdm
from torch.utils.data import Dataset
import random


columns = 'frame_id track_id class_name truncated occluded alpha 2d_x1 2d_y1 2d_x2 2d_y2 height width length x y z roty'.split()
classes = 'Car Van Truck Pedestrian Person_sitting Cyclist Tram Misc DontCare'.split()
ind_map = {}
for i,v in enumerate(columns):
    ind_map.update({v:i})

# Label_raw_data = namedtuple('Label_raw_data',
#                             'frame_id, track_id, class_name, truncated, ' + 
#                             'occluded, alpha, 2d_x1, 2d_y1, 2d_x2, 2d_y2, ' +
#                             'height, width, length, x, y, z, roty')

class One_Object(object):
    def __init__(self, frame_id, track_id, class_name, h, w, l, x, y, z, roty):
        self.frame_id = frame_id
        self.track_id = track_id
        self.class_name = class_name
        self.dimension = [l, h, w]
        self.center = [x, y, z]
        self.heading = roty
        self.next_frame_pair = None
        self.prev_frame_pair = None
        self.bbox3d = np.array([x,y,z,l,h,w,roty], dtype=np.float32)
        self.T_box2cam = kitti_utils.get_T_box2cam(np.array([x,y,z,1.]),roty)
        self.corners_4d = (np.array([[l/2,  0, w/2, 1], [l/2,  0, -w/2, 1], [-l/2,  0, -w/2, 1], [-l/2,  0, w/2, 1],
                                     [l/2, -h, w/2, 1], [l/2, -h, -w/2, 1], [-l/2, -h, -w/2, 1], [-l/2, -h, w/2, 1]]) @ self.T_box2cam.T)
        self.points = None # camera coords
        self.index = None # Large data
        self.fo_flow = None
        self.bk_flow = None

    def __repr__(self):
        out ={'first': [self.class_name, self.frame_id, self.track_id, self.center], 
              'next':  [self.next_frame_pair.class_name, self.next_frame_pair.frame_id, self.next_frame_pair.track_id, self.next_frame_pair.center] if self.next_frame_pair is not None else None}
        return str(out)


class KittiSFdataset(Dataset):
    def __init__(self, cfg, data_root, npoints=16384, is_random=True, root = '', partition='train'):
        self.data_root = data_root
        self.calib_path = os.path.join(data_root, 'calib')
        self.label_path = os.path.join(data_root, 'label_02')
        self.oxts_path  = os.path.join(data_root, 'oxts')
        # self.raw_data_path  = os.path.join(data_root, 'velodyne')
        self.data_path  = os.path.join(data_root, 'crop_velodyne')
        self.scene_path_list = sorted(glob.glob(os.path.join(self.data_path, '00*')))
        self.scene_num  = len(self.scene_path_list)
        self.frame_num_list = []
        self.K_list, self.T_cam2cam_list, self.T_lidar2cam_list, self.T_imu2lidar_list = [], [], [], []
        self.calib_para_list = []
        self.oxts_list = [] # Include IMU-GPS's raw data
        self.T_world2imu_list = []
        self.label_list = [] # Large data
        self.bg_index_list = []

        self.calib_load()
        self.oxts_load()
        self.label_load()
        self.load_obj()
        self.tot_frames = sum(self.frame_num_list)
        print("TOTAL_FRAMES", self.tot_frames)

        self.npoints = npoints
        self.is_random = is_random

        # load abandon info
        if os.path.exists(os.path.join(data_root, 'abandon_info.npy')):
            self.abandon_info = np.load(os.path.join(data_root, 'abandon_info.npy'))[1:, :] # 2, m
            self.abandon_list = [(self.abandon_info[0, i], self.abandon_info[1, i]) for i in range(self.abandon_info.shape[-1])]
            print("some frames have been moved according to abandon info")
        else:
            self.abandon_list = None


    def calib_load(self):
        calib_path_list = sorted(glob.glob(os.path.join(self.calib_path, '*.txt')))
        if len(calib_path_list) != self.scene_num:
            raise ValueError('Number of calib files is different with scenes!')
        for file_path in calib_path_list:
            with open(file_path,'r') as f:
                lines = f.readlines()
            P2, R_rect, Tr_velo_cam, Tr_imu_velo = lines[2], lines[4], lines[5], lines[6]

            P2 = np.array([float(_) for _ in P2.split()[1:13]], dtype=np.float32).reshape(3,4)
            K = P2[:,:3]
            
            R_rect = np.array([float(_) for _ in R_rect.split()[1:10]], dtype=np.float32).reshape(3,3)
            R_rect = np.concatenate((np.concatenate((R_rect,np.zeros((3,1))),axis=1),np.array([[0.,0.,0.,1.]])), axis=0)

            T_lidar2cam = np.array([float(_) for _ in Tr_velo_cam.split()[1:13]], dtype=np.float32).reshape(3,4)
            T_lidar2cam = np.concatenate((T_lidar2cam,np.array([[0.,0.,0.,1.]])), axis=0)

            T_imu2lidar = np.array([float(_) for _ in Tr_imu_velo.split()[1:13]], dtype=np.float32).reshape(3,4)
            T_imu2lidar = np.concatenate((T_imu2lidar,np.array([[0.,0.,0.,1.]])), axis=0)

            T_cam2cam = inv(P2[:,:3]) @ P2 @ R_rect
            T_cam2cam = np.concatenate((T_cam2cam,np.array([[0.,0.,0.,1.]])), axis=0)

            self.K_list.append(K)
            self.T_cam2cam_list.append(T_cam2cam) 
            self.T_lidar2cam_list.append(T_cam2cam @ T_lidar2cam) # Note: T_lidar2cam has included T_cam0_to_cam2!!!
            # self.T_lidar2cam_list.append(T_lidar2cam)
            self.T_imu2lidar_list.append(T_imu2lidar)

            self.calib_para_list.append({'K':K, 'T_cam2cam':T_cam2cam, 'T_lidar2cam':T_cam2cam @ T_lidar2cam, 'T_imu2lidar':T_imu2lidar})
            # save lidar2cam_matrix
            # lidar2cam_matrix = self.calib_para_list[0]['T_lidar2cam']
            # np.save('lidar2cam_matrix.npy', lidar2cam_matrix)
            # print("saved!")
            # self.calib_para_list.append({'K':K, 'T_cam2cam':T_cam2cam, 'T_lidar2cam':T_lidar2cam, 'T_imu2lidar':T_imu2lidar})

        print('{} calibration files have been loaded!'.format(len(self.K_list)))

    def oxts_load(self):
        oxts_path_list = sorted(glob.glob(os.path.join(self.oxts_path, '*.txt')))
        if len(oxts_path_list) != self.scene_num:
            raise ValueError('Number of oxts files is different with scenes!')
        self.oxts_list = kitti_oxts.load_oxts_packets_and_poses(oxts_path_list)

        for oxts in self.oxts_list:
            self.T_world2imu_list.append([oxt.T_world2imu for oxt in oxts])
        
        print('{} oxts files have been loaded!'.format(len(self.oxts_list)))

    def label_load(self):
        label_path_list = sorted(glob.glob(os.path.join(self.label_path, '*.txt')))
        if len(label_path_list) != self.scene_num:
            raise ValueError('Number of label files is different with scenes!')

        print('loading labels...')
        for scene_id, label_path in tqdm(enumerate(label_path_list), total=len(label_path_list), ncols=100):
            with open(label_path, 'r') as f:
                lines = f.readlines()
            frame_num = int((lines[-1].split())[0]) + 1
            raw_data_file_num = len(glob.glob(os.path.join(self.data_path, '{:04d}'.format(scene_id), '*.bin')))
            if frame_num != raw_data_file_num:
                raise ValueError('Number of frame(label) is different with data files in label_02/{:04d}.txt!'.format(scene_id))
            self.frame_num_list.append(frame_num)
            scene_label_list = [[] for _ in range(frame_num)]

            for line in lines:
                obj = line.split()
                for i, v in enumerate(obj):
                    if i == 2:
                        continue
                    if i in [0,1,3,4]:
                        obj[i] = int(v)
                    else:
                        obj[i] = float(v)
                if obj[2] != 'DontCare':
                    frame_id, track_id = obj[0], obj[1]
                    # label_raw_data = Label_raw_data(*obj)
                    # current_obj = One_Object(label_raw_data.frame_id, label_raw_data.track_id, label_raw_data.class_name,
                    #                         label_raw_data.height, label_raw_data.width, label_raw_data.length,
                    #                         label_raw_data.x, label_raw_data.y, label_raw_data.z,
                    #                         label_raw_data.roty)

                    current_obj = One_Object(obj[ind_map['frame_id']], obj[ind_map['track_id']], obj[ind_map['class_name']],
                        obj[ind_map['height']], obj[ind_map['width']], obj[ind_map['length']],
                        obj[ind_map['x']], obj[ind_map['y']], obj[ind_map['z']],
                        obj[ind_map['roty']]) 
                    scene_label_list[frame_id].append(current_obj)

                    if frame_id > 0:
                        for last_obj in scene_label_list[frame_id-1]:
                            if track_id == last_obj.track_id:
                                last_obj.next_frame_pair = current_obj
                                current_obj.prev_frame_pair = last_obj

            self.label_list.append(scene_label_list)
        print('{} label files have been loaded!'.format(len(self.label_list)))

    def load_obj(self):
        ### find forground points
        print('Finding dynamic objects\' points...')
        for scene_id, scene_all_label in tqdm(enumerate(self.label_list), total=len(self.label_list), ncols=100):
            scene_bg_index_list = [[] for _ in range(self.frame_num_list[scene_id])]
            for frame_id, frame_all_label in enumerate(scene_all_label):
                pc_lidar = np.fromfile(os.path.join(self.data_path,'{:04d}/{:06d}.bin'.format(scene_id, frame_id)), dtype=np.float32).reshape(-1,4)
                pc_lidar[:,3] = np.ones((pc_lidar.shape[0]), dtype=np.float32)
                pc_cam = pc_lidar @ self.T_lidar2cam_list[scene_id].T
                box3d_list = [obj.bbox3d for obj in frame_all_label]
                box3d = np.stack(box3d_list, axis=0) if len(box3d_list) > 0 else np.zeros([0,7]) # [M,7]
                obj_pc_list, obj_index_list = kitti_utils.find_points_in_bboxes3d(pc_cam, box3d, zaxis=1)
                
                for i, obj in enumerate(frame_all_label):
                    obj.points = obj_pc_list[i]
                    obj.index = obj_index_list[i]
                
                if obj_index_list != []:
                    bk_ind_list = list(map(np.logical_not, obj_index_list))
                    bg_index = True
                    for x in bk_ind_list:
                        bg_index *= x
                else:
                    bg_index = np.array([True]*pc_cam.shape[0])
                scene_bg_index_list[frame_id] = bg_index
            self.bg_index_list.append(scene_bg_index_list)
        
        ### compute forground flow
        print('Computing dynamic objects\' gt flow...')
        for scene_id, scene_all_label in tqdm(enumerate(self.label_list), total=len(self.label_list), ncols=100):
            for frame_id, frame_all_label in enumerate(scene_all_label):
                for i, obj in enumerate(frame_all_label):
                    if obj.next_frame_pair is not None:
                        pc1_box = obj.points @ inv(obj.T_box2cam).T
                        pc2_box = obj.next_frame_pair.points @ inv(obj.next_frame_pair.T_box2cam).T
                        obj.fo_flow, obj.bk_flow = \
                            kitti_utils.compute_gt(obj.points, obj.next_frame_pair.points, 
                                                pc1_box, pc2_box,
                                                obj.T_box2cam, obj.next_frame_pair.T_box2cam) # compute gt flow
        
        print('Object have been loaded!')
    
    def create_data(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for index in tqdm(range(self.__len__())):
            scene_id, frame_id = self.pasrse_index(index)
            if self.check_invalid_frame(scene_id, frame_id):
                continue
            pos1, pos2, color1, color2, gt, mask, fg_index, _ = self.__getitem__(index, normalize=False)
            file_path = os.path.join(save_path, '{:04d}'.format(scene_id))
            if not os.path.exists(file_path):
                os.makedirs(file_path)
            np.savez(os.path.join(file_path, '{:06d}.npz'.format(frame_id)), pc1=pos1, pc2=pos2, gt=gt, fg_index=fg_index)


    def __getitem__(self, index, normalize=True):
        scene_id, frame_id = self.pasrse_index(index)
        while self.check_invalid_frame(scene_id, frame_id):
            scene_id, frame_id = self.pasrse_index(random.randint(0, self.tot_frames-1))
        
        frame_all_label = self.label_list[scene_id][frame_id]
        pc_lidar = np.fromfile(os.path.join(self.data_path,'{:04d}/{:06d}.bin'.format(scene_id, frame_id)), dtype=np.float32).reshape(-1,4)
        pc_lidar[:,3] = np.ones((pc_lidar.shape[0]), dtype=np.float32)
        pc_cam = pc_lidar @ self.T_lidar2cam_list[scene_id].T

        ## compute bg flow

        # box3d_list = [obj.bbox3d for obj in frame_all_label]
        # box3d = np.stack(box3d_list, axis=0) if len(box3d_list) > 0 else np.zeros([0,7]) # [M,7]
        # obj_pc_list, obj_index_list = kitti_utils.find_points_in_bboxes3d(pc_cam, box3d, zaxis=1)
        # if obj_index_list != []:
        #     bk_ind_list = list(map(np.logical_not, obj_index_list))
        #     bg_index = True
        #     for x in bk_ind_list:
        #         bg_index *= x
        # else:
        #     bg_index = np.array([True]*pc_cam.shape[0])
        
        bg_index = self.bg_index_list[scene_id][frame_id]
        bg_pc_cam = pc_cam[bg_index] #[N,4]
        T_lidar2imu = inv(self.T_imu2lidar_list[scene_id])
        T_cam2lidar = inv(self.T_lidar2cam_list[scene_id])
        T1_world2imu = self.T_world2imu_list[scene_id][frame_id]
        T2_world2imu = self.T_world2imu_list[scene_id][frame_id+1]

        bg_pc_imu = bg_pc_cam @ (T_cam2lidar.T @ T_lidar2imu.T)
        bg_pc_imu_gt = bg_pc_imu @ (T1_world2imu.T @ inv(T2_world2imu).T)
        bg_pc_cam_gt = bg_pc_imu_gt @ (inv(T_lidar2imu).T @ inv(T_cam2lidar).T)
        bg_flow = (bg_pc_cam_gt - bg_pc_cam)[:,:3]
        
        ### get forground flow
        fg_pc_cam = []
        fg_flow = []
        for i, obj in enumerate(frame_all_label):
            if obj.fo_flow is None:
                continue
            fg_pc_cam.append(obj.points) #[M,4]
            fg_flow.append(obj.fo_flow[:,:3]) #[M,3]
        fg_pc_cam = np.concatenate(fg_pc_cam, axis=0) if len(fg_pc_cam) > 0 else np.zeros([0,4]) 
        fg_flow = np.concatenate(fg_flow, axis=0) if len(fg_flow) > 0 else np.zeros([0,3])
        
        all_pc = np.concatenate([bg_pc_cam, fg_pc_cam], axis=0) if fg_pc_cam.shape[0] > 0 else bg_pc_cam
        gt_flow = np.concatenate([bg_flow, fg_flow], axis=0) if fg_flow.shape[0] > 0 else bg_flow
        fg_index = np.array([False]*bg_pc_cam.shape[0] + [True]*fg_pc_cam.shape[0]) 

        pc_next_lidar = np.fromfile(os.path.join(self.data_path,'{:04d}/{:06d}.bin'.format(scene_id, frame_id+1)), dtype=np.float32).reshape(-1,4)
        pc_next_lidar[:,3] = np.ones((pc_next_lidar.shape[0]), dtype=np.float32)
        pc_next_cam = pc_next_lidar @ self.T_lidar2cam_list[scene_id].T

        pos1 = all_pc[:,:3]
        pos2 = pc_next_cam[:,:3]
        gt = gt_flow

        # random select
        n1 = pos1.shape[0]
        n2 = pos2.shape[0]
        if n1 >= self.npoints:
            if self.is_random:
                sample_idx1 = np.random.choice(n1, self.npoints, replace=False)
                pos1_ = np.copy(pos1)[sample_idx1, :]
                gt_ = np.copy(gt)[sample_idx1, :]
                fg_index = fg_index[sample_idx1]
            else:
                pos1_ = np.copy(pos1)[:self.npoints, :]
                gt_ = np.copy(gt)[:self.npoints, :]
                fg_index = fg_index[:self.npoints]
        else:
            sample_idx1 = np.concatenate((np.arange(n1), np.random.choice(n1, self.npoints-n1, replace=True)), axis=-1)
            pos1_ = np.copy(pos1)[sample_idx1, :]
            gt_ = np.copy(gt)[sample_idx1, :]
            fg_index = fg_index[sample_idx1]

        if n2 >= self.npoints:
            if self.is_random:
                sample_idx2 = np.random.choice(n2, self.npoints, replace=False)
                pos2_ = np.copy(pos2)[sample_idx2, :]
            else:
                pos2_ = np.copy(pos2)[:self.npoints, :]
        else:
            sample_idx2 = np.concatenate((np.arange(n2), np.random.choice(n2, self.npoints-n2, replace=True)), axis=-1)
            pos2_ = np.copy(pos2)[sample_idx2, :]

        if normalize:
            pos1_center = np.mean(pos1_, 0)
            pos1_ -= pos1_center
            pos2_ -= pos1_center

        color1 = np.zeros([self.npoints, 3],dtype=np.float32)
        color2 = np.zeros([self.npoints, 3],dtype=np.float32)
        mask = np.ones([self.npoints],dtype=np.float32)

        return pos1_.astype(np.float32), pos2_.astype(np.float32), color1, color2, gt_.astype(np.float32), mask, fg_index, [] # fg_index_t is None
    
    def pasrse_index(self, index):
        scene_id , frame_id = 0, 0
        # num_frames = [154, 447, 233, 144, 314, 297, 270, 800, 390, 803, 294, 373, 78, 340, 106, 376, 209, 145, 339, 1059, 837]
        num_frames = self.frame_num_list
        total_frames = sum(num_frames)
        assert index < total_frames
        slices = [0]*len(num_frames) # ignored_index
        for i, frame_num in enumerate(num_frames):
            slices[i] = slices[i-1] + frame_num if i > 0 else frame_num - 1
        while index in slices:
            index = random.randint(0, total_frames-1)
        bins = [s > index for s in slices]
        for i, flag in enumerate(bins):
            if flag:
                scene_id = i
                frame_id = index - slices[i-1] - 1 if i > 0 else index
                break
        return scene_id, frame_id
    
    def check_invalid_frame(self, scene_id, frame_id):
        if self.abandon_list is not None:
            drop_len = 10
            for i in range(drop_len // 2):
                if (scene_id, frame_id + i) in self.abandon_list or (scene_id, frame_id - i) in self.abandon_list:
                    # print(f'====== invalid scene id {scene_id}, frame id {frame_id}, drop {i} ======')
                    return True
            return False
        else:
            return False

    def __len__(self):
        return self.tot_frames