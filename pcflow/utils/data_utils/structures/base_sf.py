import numpy as np
import torch


class BaseSceneFlow(object):
    '''Base data structure for scene flow.
    '''
    def __init__(self,
                 pc1,
                 pc2,
                 gt,
                 mask1=None,
                 mask2=None,
                 height_axis=2):
        if isinstance(pc1, torch.Tensor):
            device = pc1.device
        else:
            device = torch.device('cpu')
        self.pc1 = torch.as_tensor(pc1, dtype=torch.float32, device=device)
        self.pc2 = torch.as_tensor(pc2, dtype=torch.float32, device=device)
        self.gt = torch.as_tensor(gt, dtype=torch.float32, device=device)
        if mask1 is None and mask2 is None:
            self.with_mask = False
            self.mask1, self.mask2 = None, None
        else:
            self.with_mask = True
            self.mask1 = torch.as_tensor(mask1, device=device) # may not be bool
            self.mask2 = torch.as_tensor(mask2, device=device)
        self.height_axis = height_axis
    
    def filter_points(self, idx1=None, idx2=None):
        original_type = type(self)
        if idx1 is None and idx2 is None:
            return original_type(
                self.pc1[self.mask1],
                self.pc2[self.mask2],
                self.gt[self.mask1])
        else:
            if not self.with_mask:
                return original_type(
                    self.pc1[idx1],
                    self.pc2[idx2],
                    self.gt[idx1])
            else:
                return original_type(
                    self.pc1[idx1],
                    self.pc2[idx2],
                    self.gt[idx1],
                    self.mask1[idx1],
                    self.mask2[idx2])
    
    def shuffle(self):
        idx1 = torch.randperm(self.pc1.__len__(), device=self.pc1.device)
        idx2 = torch.randperm(self.pc2.__len__(), device=self.pc1.device)
        self.pc1 = self.pc1[idx1]
        self.pc2 = self.pc2[idx2]
        self.gt = self.gt[idx1]
        if self.with_mask:
            self.mask1 = self.mask1[idx1]
            self.mask2 = self.mask1[idx2]
    
    def sample(self, num_points, replace=True, mode='random', custom_sampler=None, **kwargs):
        '''Sample the scene flow points.

        Args:
            num_points (int): number of samples whem mode is random.
            mode (str optional): random or custom.
        '''
        if mode == 'random':
            idx1 = torch.randperm(self.pc1.__len__(), device=self.pc1.device)[:num_points]
            idx2 = torch.randperm(self.pc2.__len__(), device=self.pc1.device)[:num_points]
        elif mode == 'custom':
            idx1 = custom_sampler(self.pc1, num_points, **kwargs)
            idx2 = custom_sampler(self.pc2, num_points, **kwargs)
        else:
            raise NotImplementedError
        
        if replace:
            self.pc1 = self.pc1[idx1]
            self.pc2 = self.pc2[idx2]
            self.gt = self.gt[idx1]
            if self.with_mask:
                self.mask1 = self.mask1[idx1]
                self.mask2 = self.mask1[idx2]
            return self
        else:
            original_type = type(self)
            if not self.with_mask:
                return original_type(
                    self.pc1[idx1],
                    self.pc2[idx2],
                    self.gt[idx1])
            else:
                return original_type(
                    self.pc1[idx1],
                    self.pc2[idx2],
                    self.gt[idx1],
                    self.mask1[idx1],
                    self.mask2[idx2])

    def flip(self, bev_direction='horizontal'):
        """Flip the scene flow points in BEV along given BEV direction."""
        if bev_direction == 'horizontal':
            self.pc1[:, 1] = -self.pc1[:, 1]
            self.pc2[:, 1] = -self.pc2[:, 1]
            self.gt[:, 1] = -self.gt[:, 1]
        elif bev_direction == 'vertical':
            self.pc1[:, 0] = -self.pc1[:, 0]
            self.pc2[:, 0] = -self.pc2[:, 0]
            self.gt[:, 0] = -self.gt[:, 0]

    def rotate(self, rotation, axis=None):
        '''Points rotate counterclockwise around the axis.

        Args:
            rotation (np.ndarray, torch.Tensor): int or 3x3.
        '''
        if axis is None:
            axis = self.height_axis
        if not isinstance(rotation, torch.Tensor):
            rotation = self.pc1.new_tensor(rotation)
        
        assert rotation.shape == torch.Size([3, 3]) or \
            rotation.numel() == 1, f'invalid rotation shape {rotation.shape}'

        if rotation.numel() == 1:
            rot_sin = torch.sin(rotation)
            rot_cos = torch.cos(rotation)
            if axis == 1:
                rot_mat = rotation.new_tensor([[rot_cos, 0, -rot_sin],
                                                 [0, 1, 0],
                                                 [rot_sin, 0, rot_cos]])
            elif axis == 2 or axis == -1:
                rot_mat = rotation.new_tensor([[rot_cos, -rot_sin, 0],
                                                 [rot_sin, rot_cos, 0],
                                                 [0, 0, 1]])
            elif axis == 0:
                rot_mat = rotation.new_tensor([[0, rot_cos, -rot_sin],
                                                 [0, rot_sin, rot_cos],
                                                 [1, 0, 0]])
            else:
                raise ValueError('axis should in range')
        elif rotation.numel() == 9:
            rot_mat = rotation
        else:
            raise NotImplementedError
        self.pc1[:, :3] = self.pc1[:, :3] @ rot_mat.T
        self.pc2[:, :3] = self.pc2[:, :3] @ rot_mat.T
        self.gt = self.gt @ rot_mat.T 
        return rot_mat

    def translate(self, trans_vector1, trans_vector2=None):
        """Translate scene flow points.

        Args:
            trans_vector (np.ndarray, torch.Tensor): Translation
                vector of size 3.
        """
        if trans_vector2 is None:
            if not isinstance(trans_vector1, torch.Tensor):
                trans_vector1 = self.pc1.new_tensor(trans_vector1)
            self.pc1[:, :3] += trans_vector1
            self.pc2[:, :3] += trans_vector1
        else:
            if not isinstance(trans_vector1, torch.Tensor):
                trans_vector1 = self.pc1.new_tensor(trans_vector1)
            if not isinstance(trans_vector2, torch.Tensor):
                trans_vector2 = self.pc1.new_tensor(trans_vector2)
            self.pc1[:, :3] += trans_vector1
            self.pc2[:, :3] += trans_vector2
            self.gt += trans_vector2 - trans_vector1
    
    def in_range_3d(self, point_range):
        """Check whether the points are in the given range.

        Args:
            point_range (list | torch.Tensor): The range of point
                (x_min, y_min, z_min, x_max, y_max, z_max)

        """
        in_range_mask1 = ((self.pc1[:, 0] > point_range[0])
                          & (self.pc1[:, 1] > point_range[1])
                          & (self.pc1[:, 2] > point_range[2])
                          & (self.pc1[:, 0] < point_range[3])
                          & (self.pc1[:, 1] < point_range[4])
                          & (self.pc1[:, 2] < point_range[5]))
        in_range_mask2 = ((self.pc2[:, 0] > point_range[0])
                          & (self.pc2[:, 1] > point_range[1])
                          & (self.pc2[:, 2] > point_range[2])
                          & (self.pc2[:, 0] < point_range[3])
                          & (self.pc2[:, 1] < point_range[4])
                          & (self.pc2[:, 2] < point_range[5]))
        self.pc1 = self.pc1[in_range_mask1]
        self.pc2 = self.pc2[in_range_mask2]
        self.gt = self.gt[in_range_mask1]
        if self.with_mask:
            self.mask1 = self.mask1[in_range_mask1]
            self.mask2 = self.mask2[in_range_mask2]
    
    def scale(self, scale_factor):
        """Scale the points with horizontal and vertical scaling factors.

        Args:
            scale_factors (float): Scale factors to scale the points.
        """
        self.pc1[:, :3] *= scale_factor
        self.pc2[:, :3] *= scale_factor
        self.gt[:, :3] *= scale_factor
    
    def __repr__(self):
        """str: Return a strings that describes the object."""
        return self.__class__.__name__ + '(\n    ' + str(self.tensor) + ')'

    @classmethod
    def cat(cls, sf_list):
        """Concatenate a list of SF.

        Args:
            sf_list (list[:obj:`BaseSceneFlow`]): List of SF.

        Returns:
            :obj:`BaseSceneFlow`: The concatenated SF.
        """
        assert isinstance(sf_list, (list, tuple))
        assert len(sf_list) > 0
        assert all(isinstance(sf, cls) for sf in sf_list)

        # use torch.cat (v.s. layers.cat)
        # so the returned points never share storage with input
        pc1, pc2, gt = [], [], []
        for sf in sf_list:
            pc1.append(sf.pc1)
            pc2.append(sf.pc2)
            gt.append(sf.gt)
        pc1 = torch.cat(pc1, dim=0)
        pc2 = torch.cat(pc2, dim=0)
        gt = torch.cat(gt, dim=0)
        if all(sf.with_mask for sf in sf_list):
            mask1 = torch.cat([sf.mask1 for sf in sf_list], dim=0)
            mask2 = torch.cat([sf.mask2 for sf in sf_list], dim=0)
            cat_sf = cls(pc1, pc2, gt, mask1, mask2)
        else:
            cat_sf = cls(pc1, pc2, gt)
        return cat_sf
    
    @classmethod
    def stack(cls, sf_list):
        """Stack a batch of SF.

        Args:
            sf_list (list[:obj:`BaseSceneFlow`]): List of SF (a batch).

        """
        assert isinstance(sf_list, (list, tuple))
        assert len(sf_list) > 0
        assert all(isinstance(sf, cls) for sf in sf_list)

        # use torch.cat (v.s. layers.cat)
        # so the returned points never share storage with input
        pc1, pc2, gt = [], [], []
        for sf in sf_list:
            pc1.append(sf.pc1)
            pc2.append(sf.pc2)
            gt.append(sf.gt)
        pc1 = torch.stack(pc1, dim=0)
        pc2 = torch.stack(pc2, dim=0)
        gt = torch.stack(gt, dim=0)
        if all(sf.with_mask for sf in sf_list):
            mask1 = torch.stack([sf.mask1 for sf in sf_list], dim=0)
            mask2 = torch.stack([sf.mask2 for sf in sf_list], dim=0)
            return pc1, pc2, gt, mask1, mask2
        else:
            return pc1, pc2, gt
    
    def to(self, device):
        """Convert current scene flow to a specific device.

        Args:
            device (str | :obj:`torch.device`): The name of the device.

        Returns:
            :obj:`BaseSceneFlow`: A new scene flow object on the \
                specific device.
        """
        if self.with_mask:
            mask1 = self.mask1.to(device)
            mask2 = self.mask2.to(device)
        else:
            mask1, mask2 = None, None
        original_type = type(self)
        return original_type(
            self.pc1.to(device),
            self.pc2.to(device),
            self.gt.to(device),
            mask1, mask2)

    def clone(self):
        """Clone the scene flow object.

        """
        if self.with_mask:
            mask1 = self.mask1.clone()
            mask2 = self.mask2.clone()
        else:
            mask1, mask2 = None, None
        original_type = type(self)
        return original_type(
            self.pc1.clone(),
            self.pc2.clone(),
            self.gt.clone(),
            mask1, mask2)

    @property
    def device(self):
        """str: The device of the points are on."""
        return self.pc1.device
    
    @property
    def data(self):
        """str: The device of the points are on."""
        return self.pc1, self.pc2, self.gt, self.mask1, self.mask2
