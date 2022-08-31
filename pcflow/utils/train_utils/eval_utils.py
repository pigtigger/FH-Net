import torch
import torch.distributed as dist


def scene_flow_EPE(pred, labels, mask=None):
    '''
    Args:
        pred: b x n x 3
        lables: b x n x 3
        mask: b x n
    '''
    if mask is None:
        b, n, _ = pred.shape
        mask = torch.ones((b,n), dtype=bool)
    error = torch.sqrt(torch.sum((pred - labels)**2, 2) + 1e-20)

    gtflow_len = torch.sqrt(torch.sum(labels*labels, 2) + 1e-20) # B,N
    acc1 = torch.sum(torch.logical_or((error <= 0.05)*mask, (error/gtflow_len <= 0.05)*mask), 1)
    acc2 = torch.sum(torch.logical_or((error <= 0.1)*mask, (error/gtflow_len <= 0.1)*mask), 1)
    outlier = torch.sum(torch.logical_or((error > 0.3)*mask, (error/gtflow_len > 0.1)*mask), 1)

    mask_sum = torch.sum(mask, 1)
    acc1 = acc1[mask_sum > 0] / mask_sum[mask_sum > 0]
    acc1 = torch.mean(acc1)
    acc2 = acc2[mask_sum > 0] / mask_sum[mask_sum > 0]
    acc2 = torch.mean(acc2)
    outlier = outlier[mask_sum > 0] / mask_sum[mask_sum > 0]
    outlier = torch.mean(outlier)

    epe = torch.sum(error * mask, 1)[mask_sum > 0] / mask_sum[mask_sum > 0]
    epe = torch.mean(epe)
    return epe, acc1, acc2, outlier