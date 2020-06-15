epsilon = 1e-16

def to_cpu(tensor):
    return tensor.detach().cpu()

def build_targets(pred_boxes, pred_cls, target, anchors, ignore_threshold):
    """
    :param pred_boxes:  (batch, num_anchors, width, height, 4)
    :param pred_cls:    (batch, num_anchors, width, height, num_classes)
    :param target:      (num_ground_true_bboxes, 6)
    :param anchors:     (num_anchors, 2)
    :param ignore_threshold:
    :return:
    """

    ByteTensor = torch.cuda.ByteTensor if pred_boxes.is_cuda else torch.ByteTensor
    FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor
    BoolTensor = torch.cuda.BoolTensor if pred_boxes.is_cuda else torch.BoolTensor

    batch_size = pred_boxes.size(0)
    num_anchors = pred_boxes.size(1)
    num_classes = pred_cls.size(-1)
    grid_size = pred_boxes.size(2)

    # Indicate which anchor captured obj.
    obj_mask = BoolTensor(batch_size, num_anchors, grid_size, grid_size).fill_(0)

    # Indicate anchor that not captured obj.
    noobj_mask = BoolTensor(batch_size, num_anchors, grid_size, grid_size).fill_(1)

    class_mask = FloatTensor(batch_size, num_anchors, grid_size, grid_size).fill_(0)
    iou_scores = FloatTensor(batch_size, num_anchors, grid_size, grid_size).fill_(0)

    tboxes = FloatTensor(batch_size, num_anchors, grid_size, grid_size, 4).fill_(0)
    tcls = FloatTensor(batch_size, num_anchors, grid_size, grid_size, num_classes).fill_(0)

    # Convert to position relative to box
    target_boxes = target[:, 2:6] * grid_size  # (n, 4)
    gxy = target_boxes[:, :2]  # (n, 2)
    gwh = target_boxes[:, 2:]  # (n, 2)

    # Find anchors with best iou with targets from all anchors
    ious = bbox_wh_iou(anchors, gwh)  # (num_anchors, n)
    best_ious, best_idx = ious.max(0)  # (n, ), (n, )

    # Separate target values
    b, target_labels = target[:, :2].long().t()  # (n, ), (n, )
    gi, gj = gxy.long().t()

    # Sets masks
    obj_mask[b, best_idx, gj, gi] = 1
    noobj_mask[b, best_idx, gj, gi] = 0

    # Set noobj mask to zero where iou exceeds ignore threshold
    # for i, anchor_ious in enumerate(ious.t()):
    #     noobj_mask[b[i], anchor_ious > ignore_threshold, gj[i], gi[i]] = 0
    noobj_mask[b, :, gj, gi] = noobj_mask[b, :, gj, gi] * (ious.t() <= ignore_threshold)

    tboxes[b, best_idx, gj, gi, 0:2] = gxy - gxy.floor()
    tboxes[b, best_idx, gj, gi, 2:4] = torch.log(gwh / anchors[best_idx] + epsilon)

    # One-hot encoding of label
    tcls[b, best_idx, gj, gi, target_labels] = 1

    # Compute label correctness and iou at best anchor
    class_mask[b, best_idx, gj, gi] = (pred_cls[b, best_idx, gj, gi].argmax(-1) == target_labels).float()
    iou_scores[b, best_idx, gj, gi] = bbox_iou(pred_boxes[b, best_idx, gj, gi], target_boxes, p1p2=False)

    tconf = obj_mask.float()
    return iou_scores, class_mask, obj_mask, noobj_mask, tboxes, tcls, tconf

def parse_model_config(path):
    """Parses the yolo-v3 layer configuration file and returns module definitions"""
    file = open(path, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines]  # get rid of fringe whitespaces
    module_defs = []
    for line in lines:
        if line.startswith('['):  # This marks the start of a new block
            module_defs.append({})
            module_defs[-1]['type'] = line[1:-1].rstrip()
            if module_defs[-1]['type'] == 'convolutional':
                module_defs[-1]['batch_normalize'] = 0
        else:
            key, value = line.split("=")
            value = value.strip()
            module_defs[-1][key.rstrip()] = value.strip()

    return module_defs