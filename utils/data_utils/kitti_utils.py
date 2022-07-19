import numpy as np
import torch
from utils.data_utils.box_np_ops import points_in_rbbox

def compute_gt(pc1_cam, pc2_cam, pc1_box, pc2_box, T1_box2cam, T2_box2cam):
    gt_pc1_cam = pc1_box @ T2_box2cam.T
    gt_flow1_2 = (gt_pc1_cam - pc1_cam)[:,:3]
    gt_pc2_cam = pc2_box @ T1_box2cam.T
    gt_flow2_1 = (gt_pc2_cam - pc2_cam)[:,:3]
    return gt_flow1_2, gt_flow2_1

@torch.no_grad()
def find_points_in_bboxes3d(in_points, in_bboxes3d, zaxis=1):
    """
    Args:
        points (tensor:N x dims)
        bboxes3d (tensor(M x 7))

    Returns:
        list[tensor: num_pts x dims] box_np_ops.points_in_rbbox(points, gt_boxes_3d) /core/bbox/..
    """
    if isinstance(in_points,torch.Tensor):
        points = in_points.cpu().numpy()
    elif isinstance(in_points,np.ndarray):
        points = in_points
    else:
        raise NotImplementedError("expect tensor or numpy but got{}".format(type(in_points)))\
    
    if isinstance(in_bboxes3d,torch.Tensor):
        bboxes3d = in_bboxes3d.cpu().numpy()
    elif isinstance(in_bboxes3d,np.ndarray):
        bboxes3d = in_bboxes3d
    else:
        raise NotImplementedError("expect tensor or numpy but got{}".format(type(in_bboxes3d)))

    num_boxes = bboxes3d.shape[0]
    point_indices = points_in_rbbox(points, bboxes3d, z_axis=zaxis, origin=(0.5,1,0.5))#[N,M]
    points_in_box_list = []
    points_inds_list = []
    for i in range(num_boxes):
        points_in = in_points[point_indices[:,i]]
        points_inds_list.append(point_indices[:,i])
        # points_in[:,:3] -= in_bboxes3d[i,:3]
        # if points_in.shape[0] != 0:
          # print("points_find_in: , bbox",points_in.shape, bboxes3d[i])
        points_in_box_list.append(points_in)
    return points_in_box_list , points_inds_list