import numpy as np


class RandomFlipSF(object):
    '''Flip the points & scene flow gt.
    '''
    def __init__(self,
                 flip_ratio_bev_horizontal=0.5,
                 flip_ratio_bev_vertical=0.5):
        self.flip_ratio_bev_horizontal = flip_ratio_bev_horizontal
        self.flip_ratio_bev_vertical = flip_ratio_bev_vertical
    
    def __call__(self, data_dict):
        sf_data = data_dict['sf_data']
        flip_horizontal = True if np.random.rand(
            ) < self.flip_ratio_bev_horizontal else False
        flip_vertical = True if np.random.rand(
            ) < self.flip_ratio_bev_vertical else False
        if flip_horizontal:
            sf_data.flip(bev_direction='horizontal')
        if flip_vertical:
            sf_data.flip(bev_direction='vertical')
    
    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f' flip_ratio_bev_horizontal={self.flip_ratio_bev_horizontal})'
        repr_str += f' flip_ratio_bev_vertical={self.flip_ratio_bev_vertical})'
        return repr_str
    

class GlobalRotScaleTrans(object):
    """Apply global rotation, scaling and translation to a 3D scene.

    Args:
        rot_range (list[float]): Range of rotation angle.
            Defaults to [-0.78539816, 0.78539816] (close to [-pi/4, pi/4]).
        scale_ratio_range (list[float]): Range of scale ratio.
            Defaults to [0.95, 1.05].
        translation_std (list[float]): The standard deviation of translation
            noise. This applies random translation to a scene by a noise, which
            is sampled from a gaussian distribution whose standard deviation
            is set by ``translation_std``. Defaults to [0, 0, 0]
        shift_height (bool): Whether to shift height.
            (the fourth dimension of indoor points) when scaling.
            Defaults to False.
    """

    def __init__(self,
                 rot_range=[-0.78539816, 0.78539816],
                 scale_ratio_range=[0.95, 1.05],
                 translation_std=[0, 0, 0],
                 shift_height=False):
        seq_types = (list, tuple, np.ndarray)
        if not isinstance(rot_range, seq_types):
            assert isinstance(rot_range, (int, float)), \
                f'unsupported rot_range type {type(rot_range)}'
            rot_range = [-rot_range, rot_range]
        self.rot_range = rot_range

        assert isinstance(scale_ratio_range, seq_types), \
            f'unsupported scale_ratio_range type {type(scale_ratio_range)}'
        self.scale_ratio_range = scale_ratio_range

        if not isinstance(translation_std, seq_types):
            assert isinstance(translation_std, (int, float)), \
                f'unsupported translation_std type {type(translation_std)}'
            translation_std = [
                translation_std, translation_std, translation_std
            ]
        assert all([std >= 0 for std in translation_std]), \
            'translation_std should be positive'
        self.translation_std = np.array(translation_std, dtype=np.float32)
        self.shift_height = shift_height

    def _trans_scene_flow(self, data_dict):
        """Private function to translate scene points and flow.

        Args:
            data_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after translation, 'sf_data', 'pcd_trans' \
                are updated in the result dict.
        """
        trans_vector = np.random.normal(scale=self.translation_std, size=3).T

        data_dict['sf_data'].translate(trans_vector)
        data_dict['pcd_trans'] = trans_vector

    def _rot_scene_flow(self, data_dict):
        """Private function to rotate bounding points and flow.

        Args:
            data_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after rotation, 'sf_data', 'pcd_rotation' \
                are updated in the result dict.
        """
        rotation = self.rot_range
        noise_rotation = np.random.uniform(rotation[0], rotation[1])

        # if no bbox in data_dict, only rotate points
        rot_mat = data_dict['points'].rotate(noise_rotation)
        data_dict['pcd_rotation'] = rot_mat

    def _scale_scene_flow(self, data_dict):
        """Private function to scale bounding points and flow.

        Args:
            data_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after scaling, 'sf_data', 'pcd_rotation' \
                are updated in the result dict.
        """
        scale_factor = np.random.uniform(self.scale_ratio_range[0],
                                         self.scale_ratio_range[1])
        data_dict['sf_data'].scale(scale_factor)
        data_dict['pcd_scale_factor'] = scale_factor

    def __call__(self, data_dict):
        """Private function to rotate, scale and translate bounding boxes and \
        points.

        Args:
            data_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after scaling, 'points', 'pcd_rotation',
                'pcd_scale_factor', 'pcd_trans' are updated \
                in the result dict.
        """
        if 'transformation_flow' not in data_dict:
            data_dict['transformation_flow'] = []

        self._rot_scene_flow(data_dict)
        self._scale_scene_flow(data_dict)
        self._trans_scene_flow(data_dict)

        data_dict['transformation_flow'].extend(['R', 'S', 'T'])
        return data_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(rot_range={self.rot_range},'
        repr_str += f' scale_ratio_range={self.scale_ratio_range},'
        repr_str += f' translation_std={self.translation_std.tolist()},'
        repr_str += f' shift_height={self.shift_height})'
        return repr_str


class PointShuffle(object):
    """Shuffle scene points."""

    def __call__(self, data_dict):
        """Call function to shuffle points.

        Args:
            data_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'sf_data' is updated in \
                the result dict.
        """
        data_dict['sf_data'].shuffle()
        return data_dict

    def __repr__(self):
        return self.__class__.__name__


class PointsRangeFilter(object):
    """Filter scene points by the range.

    Args:
        point_cloud_range (list[float]): Point cloud range.
    """

    def __init__(self, point_cloud_range):
        self.pcd_range = np.array(point_cloud_range, dtype=np.float32)

    def __call__(self, data_dict):
        """Call function to filter points by the range.

        Args:
            data_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'sf_data' is updated in \
                the result dict.
        """
        data_dict['sf_data'].in_range_3d(self.pcd_range)
        return data_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(point_cloud_range={self.pcd_range.tolist()})'
        return repr_str


class RandomPointSample(object):
    """Scene point sample.

    Sampling data to a certain number.

    Args:
        num_points (int): Number of points to be sampled.
        sample_range (float, optional): The range where to sample points.
            If not None, the points with depth larger than `sample_range` are
            prior to be sampled. Defaults to None.
        replace (bool, optional): Whether the sampling is with or without
            replacement. Defaults to False.
    """

    def __init__(self, num_points, replace=False):
        self.num_points = num_points
        self.replace = replace

    def __call__(self, data_dict):
        """Call function to sample scene points.

        Args:
            data_dict (dict): Result dict from loading pipeline.
        
        Returns:
            dict: Results after filtering, 'sf_data' is updated in \
                the result dict.
        """
        data_dict['sf_data'] = data_dict['sf_data'].sample(
            num_points=self.num_points, replace=self.replace, mode='random')
        return data_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(num_points={self.num_points},'
        repr_str += f' replace={self.replace})'
        return repr_str