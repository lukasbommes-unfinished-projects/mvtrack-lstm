import math
import torch
from torch.nn.parameter import Parameter


def bbox_transform_inv_otcd(boxes, deltas, batch_size=None, sigma=1/math.sqrt(2), add_one=True):
    if add_one:  # original OTCD implementation
        widths = boxes[:, :, 2] - boxes[:, :, 0] + 1.0
        heights = boxes[:, :, 3] - boxes[:, :, 1] + 1.0
    else: # my correction which keeps boxes constant when deltas are zero
        widths = boxes[:, :, 2] - boxes[:, :, 0]
        heights = boxes[:, :, 3] - boxes[:, :, 1]
    ctr_x = boxes[:, :, 0] + 0.5 * widths
    ctr_y = boxes[:, :, 1] + 0.5 * heights

    dx = deltas[:, :, 0::4]
    dy = deltas[:, :, 1::4]
    dw = deltas[:, :, 2::4]
    dh = deltas[:, :, 3::4]

    pred_ctr_x = dx * widths.unsqueeze(2) / (math.sqrt(2) * sigma) + ctr_x.unsqueeze(2)
    pred_ctr_y = dy * heights.unsqueeze(2) / (math.sqrt(2) * sigma) + ctr_y.unsqueeze(2)
    pred_w = torch.exp(dw) * widths.unsqueeze(2)
    pred_h = torch.exp(dh) * heights.unsqueeze(2)

    pred_boxes = deltas.clone()
    # x1
    pred_boxes[:, :, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, :, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, :, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, :, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes  # [x1, y1, x2, y2]


# def velocities_from_boxes(boxes_prev, boxes):
#     """Computes bounding box velocities.
#
#     Args:
#         boxes_prev (`torch.Tensor`): Bounding boxes in previous frame. Shape [B, N, 4]
#             where B is the batch size and N the number of bounding boxes.
#
#         boxes (`torch.Tensor`): Bounding boxes in current frame. Shape [B, N, 4]
#             where B is the batch size and N the number of bounding boxes.
#
#     Returns:
#         (`torch.Tensor`) velocities of box coordinates in current frame. Shape [B, N, 4].
#
#     Ensure that ordering of boxes in both tensors is consistent and that the number of boxes
#     is the same.
#     """
#     x = boxes[:, 0]
#     y = boxes[:, 1]
#     w = boxes[:, 2]
#     h = boxes[:, 3]
#     x_p = boxes_prev[:, 0]
#     y_p = boxes_prev[:, 1]
#     w_p = boxes_prev[:, 2]
#     h_p = boxes_prev[:, 3]
#     v_x = (1 / w_p * (x - x_p)).unsqueeze(-1)
#     v_y = (1 / h_p * (y - y_p)).unsqueeze(-1)
#     v_w = (torch.log(w / w_p)).unsqueeze(-1)
#     v_h = (torch.log(h / h_p)).unsqueeze(-1)
#     return torch.cat([v_x, v_y, v_w, v_h], -1)
#
#
# def box_from_velocities(boxes_prev, velocities):
#     """Computes bounding boxes from previous boxes and velocities.
#
#     Args:
#         boxes_prev (`torch.Tensor`): Bounding boxes in previous frame. Shape [B, N, 4]
#             where B is the batch size and N the number of bounding boxes.
#
#         velocities (`torch.Tensor`): Box velocities in current frame. Shape [B, N, 4]
#         where B is the batch size and N the number of bounding boxes.
#
#     Returns:
#         (`torch.Tensor`) Bounding boxes in current frame. Shape [B, N, 4].
#
#     Ensure that ordering of boxes and velocities in both tensors is consistent that is
#     box in row i should correspond to velocities in row i.
#     """
#     x_p = boxes_prev[:, 0]
#     y_p = boxes_prev[:, 1]
#     w_p = boxes_prev[:, 2]
#     h_p = boxes_prev[:, 3]
#     v_x = velocities[:, 0]
#     v_y = velocities[:, 1]
#     v_w = velocities[:, 2]
#     v_h = velocities[:, 3]
#     x = (w_p * v_x + x_p).unsqueeze(-1)
#     y = (h_p * v_y + y_p).unsqueeze(-1)
#     w = (w_p * torch.exp(v_w)).unsqueeze(-1)
#     h = (h_p * torch.exp(v_h)).unsqueeze(-1)
#     return torch.cat([x, y, w, h], -1)


def velocities_from_boxes(boxes_prev, boxes, sigma=1.5):
    """Computes bounding box velocities.

    Args:
        boxes_prev (`torch.Tensor`): Bounding boxes in previous frame. Shape [B, N, 4]
            where B is the batch size and N the number of bounding boxes. Each row in
            this tensor corresponds to one bounding box in format [x, y, w, h] where
            x and y are the coordinates of the top left corner and w and h are box
            width and height.

        boxes (`torch.Tensor`): Bounding boxes in current frame. Shape [B, N, 4]
            where B is the batch size and N the number of bounding boxes. Same format
            as boxes_prev.

        sigma (`float`): A constant scaling factor which is multiplied with the box center velocities.

    Returns:
        (`torch.Tensor`) velocities of box coordinates in current frame. Shape [B, N, 4].
        Each row in this tensor corresponds to a box velocity in format [v_xc, v_yc, v_w, v_h]
        where v_xc and v_yc are velocities of the box center point and v_w and v_h are
        velocities of the box width and height.

    Ensure that ordering of boxes in both tensors is consistent and that the number of boxes
    is the same.
    """
    w = boxes[:, 2]
    h = boxes[:, 3]
    xc = boxes[:, 0] + 0.5 * w
    yc = boxes[:, 1] + 0.5 * h
    w_p = boxes_prev[:, 2]
    h_p = boxes_prev[:, 3]
    xc_p = boxes_prev[:, 0] + 0.5 * w_p
    yc_p = boxes_prev[:, 1] + 0.5 * h_p
    v_xc = (math.sqrt(2) * sigma / w_p * (xc - xc_p)).unsqueeze(-1)
    v_yc = (math.sqrt(2) * sigma / h_p * (yc - yc_p)).unsqueeze(-1)
    v_w = (torch.log(w / w_p)).unsqueeze(-1)
    v_h = (torch.log(h / h_p)).unsqueeze(-1)
    return torch.cat([v_xc, v_yc, v_w, v_h], -1)


def box_from_velocities(boxes_prev, velocities, sigma=1.5):
    """Computes bounding boxes from previous boxes and velocities.

    Args:
        boxes_prev (`torch.Tensor`): Bounding boxes in previous frame. Shape [B, N, 4]
            where B is the batch size and N the number of bounding boxes.
            Each row in this tensor corresponds to one bounding box in format
            [x, y, w, h] where x and y are the coordinates of the top left
            corner and w and h are box width and height.

        velocities (`torch.Tensor`): Box velocities in current frame. Shape [B, N, 4]
            where B is the batch size and N the number of bounding boxes.
            Each row in this tensor corresponds to a box velocity in format [v_xc, v_yc, v_w, v_h]
            where v_xc and v_yc are velocities of the box center point and v_w and v_h are
            velocities of the box width and height.

        sigma (`float`): A constant scaling factor which is multiplied with the box center velocities.

    Returns:
        (`torch.Tensor`) Bounding boxes in current frame. Shape [B, N, 4]. Same format
        as boxes_prev.

    Ensure that ordering of boxes and velocities in both tensors is consistent that is
    box in row i should correspond to velocities in row i.
    """
    w_p = boxes_prev[:, 2]
    h_p = boxes_prev[:, 3]
    xc_p = boxes_prev[:, 0] + 0.5 * w_p
    yc_p = boxes_prev[:, 1] + 0.5 * h_p
    v_xc = velocities[:, 0]
    v_yc = velocities[:, 1]
    v_w = velocities[:, 2]
    v_h = velocities[:, 3]
    xc = (w_p / (math.sqrt(2) * sigma) * v_xc + xc_p).unsqueeze(-1)
    yc = (h_p / (math.sqrt(2) * sigma) * v_yc + yc_p).unsqueeze(-1)
    w = (w_p * torch.exp(v_w)).unsqueeze(-1)
    h = (h_p * torch.exp(v_h)).unsqueeze(-1)
    x = xc - 0.5 * w
    y = yc - 0.5 * h
    return torch.cat([x, y, w, h], -1)


def velocities_from_boxes_2d(boxes_prev, boxes):
    """Computes bounding box center point velocities.

    Args:
        boxes_prev (`torch.Tensor`): Bounding boxes in previous frame. Shape [N, 4]
            where N is the number of bounding boxes. Each row in this tensor corresponds
            to one bounding box in format [x, y, w, h] where x and y are the coordinates
            of the top left corner and w and h are box width and height.

        boxes (`torch.Tensor`): Bounding boxes in current frame. Shape [N, 4]
            where N is the number of bounding boxes. Same format as boxes_prev.

    Returns:
        (`torch.Tensor`) velocities of box coordinates in current frame. Shape [N, 2].
        Each row in this tensor corresponds to a box velocity in format [v_xc, v_yc]
        where v_xc and v_yc are velocities of the box center point.

    Ensure that ordering of boxes in both tensors is consistent and that the number of boxes
    is the same.
    """
    w = boxes[:, 2]
    h = boxes[:, 3]
    xc = boxes[:, 0] + 0.5 * w
    yc = boxes[:, 1] + 0.5 * h
    w_p = boxes_prev[:, 2]
    h_p = boxes_prev[:, 3]
    xc_p = boxes_prev[:, 0] + 0.5 * w_p
    yc_p = boxes_prev[:, 1] + 0.5 * h_p
    v_xc = (1 / w_p * (xc - xc_p)).unsqueeze(-1)
    v_yc = (1 / h_p * (yc - yc_p)).unsqueeze(-1)
    return torch.cat([v_xc, v_yc], -1)


def box_from_velocities_2d(boxes_prev, velocities):
    """Computes bounding boxes from previous boxes and center point velocities.

    Args:
        boxes_prev (`torch.Tensor`): Bounding boxes in previous frame. Shape [N, 4]
            where N is the number of bounding boxes. Each row in this tensor corresponds
            to one bounding box in format  [x, y, w, h] where x and y are the coordinates
            of the top left corner and w and h are box width and height.

        velocities (`torch.Tensor`): Box velocities in current frame. Shape [N, 2]
            where N is the number of bounding boxes. Each row in this tensor corresponds
            to a box velocity in format [v_xc, v_yc, v_w, v_h] where v_xc and v_yc are
            velocities of the box center point and v_w and v_h.

    Returns:
        (`torch.Tensor`) Bounding boxes in current frame. Shape [N, 4]. Same format
        as boxes_prev.

    Ensure that ordering of boxes and velocities in both tensors is consistent that is
    box in row i should correspond to velocities in row i.
    """
    w_p = boxes_prev[:, 2]
    h_p = boxes_prev[:, 3]
    xc_p = boxes_prev[:, 0] + 0.5 * w_p
    yc_p = boxes_prev[:, 1] + 0.5 * h_p
    v_xc = velocities[:, 0]
    v_yc = velocities[:, 1]
    xc = (w_p * v_xc + xc_p).unsqueeze(-1)
    yc = (h_p * v_yc + yc_p).unsqueeze(-1)
    w = w_p.unsqueeze(-1)
    h = h_p.unsqueeze(-1)
    x = xc - 0.5 * w
    y = yc - 0.5 * h
    return torch.cat([x, y, w, h], -1)


# TESTING
if __name__ == "__main__":
    import numpy as np

    gt_boxes_prev = np.array([[1338.,  418.,  167.,  379.],
              [ 586.,  447.,   85.,  263.],
              [1416.,  431.,  184.,  336.],
              [1056.,  484.,   36.,  110.],
              [1091.,  484.,   31.,  115.],
              [1255.,  447.,   33.,  100.],
              [1016.,  430.,   40.,  116.],
              [1101.,  441.,   38.,  108.],
              [ 935.,  436.,   42.,  114.],
              [ 442.,  446.,  105.,  283.],
              [ 636.,  458.,   61.,  187.],
              [1364.,  434.,   51.,  124.],
              [1478.,  434.,   63.,  124.],
              [ 473.,  460.,   89.,  249.],
              [ 548.,  465.,   35.,   93.],
              [ 418.,  459.,   40.,   84.],
              [ 582.,  456.,   35.,  133.],
              [ 972.,  456.,   32.,   77.],
              [ 578.,  432.,   20.,   43.],
              [ 596.,  429.,   18.,   42.],
              [ 663.,  451.,   34.,   86.]])

    gt_boxes = np.array([[1342.,  417.,  168.,  380.],
              [ 586.,  446.,   85.,  264.],
              [1422.,  431.,  183.,  337.],
              [1055.,  483.,   36.,  110.],
              [1090.,  484.,   32.,  114.],
              [1255.,  447.,   33.,  100.],
              [1015.,  430.,   40.,  116.],
              [1100.,  440.,   38.,  108.],
              [ 934.,  435.,   42.,  114.],
              [ 442.,  446.,  107.,  282.],
              [ 636.,  458.,   61.,  187.],
              [1365.,  434.,   52.,  124.],
              [1480.,  433.,   62.,  125.],
              [ 473.,  460.,   89.,  249.],
              [ 547.,  464.,   35.,   93.],
              [ 418.,  459.,   40.,   84.],
              [ 582.,  455.,   34.,  134.],
              [ 972.,  456.,   32.,   77.],
              [ 578.,  431.,   20.,   43.],
              [ 595.,  428.,   18.,   42.],
              [1035.,  452.,   25.,   67.],
              [ 664.,  451.,   34.,   85.]])

    gt_ids = np.array([ 2.,  3.,  8.,  9., 10., 14., 15., 17., 18., 19., 20., 21., 22., 23.,
              26., 31., 36., 39., 68., 69., 70., 72.])

    gt_ids_prev = np.array([ 2.,  3.,  8.,  9., 10., 14., 15., 17., 18., 19., 20., 21., 22., 23.,
              26., 31., 36., 39., 68., 69., 72.])

    _, idx_1, idx_0 = np.intersect1d(gt_ids, gt_ids_prev, assume_unique=True, return_indices=True)
    print(idx_1, idx_0)
    boxes = torch.from_numpy(gt_boxes[idx_1])
    boxes_prev = torch.from_numpy(gt_boxes_prev[idx_0])

    print(boxes.shape)
    print(boxes_prev.shape)

    print("boxes_prev:", boxes_prev)

    velocities = velocities_from_boxes(boxes_prev, boxes)
    print(velocities)
    print(velocities.shape)

    box = box_from_velocities(boxes_prev, velocities)
    print(box)
    print(box.shape)
    print(boxes)
