import cv2
from time import perf_counter
import numpy as np
import torch
from torchvision.ops.boxes import batched_nms

import os
THIS_DIR = os.path.dirname(os.path.realpath(__file__))
CWD = os.getcwd()
if CWD == THIS_DIR:
    from models import Darknet
else:
    from .models import Darknet


class YOLOV4(object):
    if CWD == THIS_DIR:
        _defaults = {
            "weights": "weights/yolov4.weights",
            "config": "cfg/yolov4.cfg",
            "classes_path": 'cfg/coco.names',
            "thresh": 0.5,
            "nms_thresh": 0.4,
            "model_image_size": (608,608),
            "max_batch_size": 4,
            "half": True
        }
    else:
        _defaults = {
            "weights": "yolov4_pytorch/weights/yolov4.weights",
            "config": "yolov4_pytorch/cfg/yolov4.cfg",
            "classes_path": 'yolov4_pytorch/cfg/coco.names',
            "thresh": 0.5,
            "nms_thresh": 0.4,
            "model_image_size": (608,608),
            "max_batch_size": 4,
            "half": True
        }

    def __init__(self, bgr=True, gpu_device=0, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        # for portability between keras-yolo3/yolo.py and this
        if 'model_path' in kwargs:
            kwargs['weights'] = kwargs['model_path']
        if 'score' in kwargs:
            kwargs['thresh'] = kwargs['score']
        self.__dict__.update(kwargs) # update with user overrides

        self.class_names = self._get_class()
        self.model = Darknet(self.config)
        self.model.load_darknet_weights(self.weights)

        self.device = gpu_device
        self.model.cuda(self.device)
        self.model.eval()

        self.bgr = bgr

        if self.half:
            self.model.half()

        # warm up
        self._detect([np.zeros((10,10,3), dtype=np.uint8)])
        print('Warmed up!')

    def _get_class(self):
        with open(self.classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _detect(self, list_of_imgs):
        inputs = []
        for img in list_of_imgs:
            if self.bgr:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # print('bgr: {}'.format(img.shape))
            # print('size: {}'.format(self.model_image_size))
            image = cv2.resize(img, self.model_image_size)
            # print('image: {}'.format(image.shape))
            inputs.append(np.expand_dims(np.array(image), axis=0))

        images = np.concatenate(inputs, 0)

        # print('images: {}'.format(images.shape))
        images = torch.from_numpy(images.transpose(0, 3, 1, 2)).float().div(255.0)

        images = images.cuda()
        images = torch.autograd.Variable(images)

        if self.half:
            images = images.half()

        batches = []
        for i in range(0, len(images), self.max_batch_size):
            these_imgs = images[i:i+self.max_batch_size]
            batches.append(these_imgs)

        feature_list = None
        with torch.no_grad():
            for batch in batches:
                img = batch.cuda(self.device)
                features = self.model(img)

                if feature_list is None:
                    feature_list = features
                else:
                    feature_list = torch.cat((feature_list, features))

        # feature_list: (batch, height * width * num_anchors, 5 + num_classes)
        return feature_list

    def detect_get_box_in(self, images, box_format='ltrb', classes=None, buffer_ratio=0.0):
        '''
        Params
        ------
        - images : ndarray-like or list of ndarray-like
        - box_format : string of characters representing format order, where l = left, t = top, r = right, b = bottom, w = width and h = height
        - classes : list of string, classes to focus on
        - buffer : float, proportion of buffer around the width and height of the bounding box

        Returns
        -------
        if one ndarray given, this returns a list (boxes in one image) of tuple (box_infos, score, predicted_class),
        
        else if a list of ndarray given, this return a list (batch) containing the former as the elements,

        where,
            - box_infos : list of floats in the given box format
            - score : float, confidence level of prediction
            - predicted_class : string

        '''
        single = False
        if isinstance(images, list):
            if len(images) <= 0 : 
                return None
            else:
                assert all(isinstance(im, np.ndarray) for im in images)
        elif isinstance(images, np.ndarray):
            images = [ images ]
            single = True

        res = self._detect(images)
        frame_shapes = [image.shape for image in images]
        all_dets = self._postprocess(res, shapes=frame_shapes, box_format=box_format, classes=classes, buffer_ratio=buffer_ratio)

        if single:
            return all_dets[0]
        else:
            return all_dets

    def get_detections_dict(self, frames, classes=None, buffer_ratio=0.0):
        '''
        Params: frames, list of ndarray-like
        Returns: detections, list of dict, whose key: label, confidence, t, l, w, h
        '''
        if frames is None or len(frames) == 0:
            return None
        all_dets = self.detect_get_box_in( frames, box_format='tlbrwh', classes=classes, buffer_ratio=buffer_ratio )
        
        all_detections = []
        for dets in all_dets:
            detections = []
            for tlbrwh,confidence,label in dets:
                top, left, bot, right, width, height = tlbrwh
                detections.append( {'label':label,'confidence':confidence,'t':top,'l':left,'b':bot,'r':right,'w':width,'h':height} )
            all_detections.append(detections)
        return all_detections

    def _nms(self, predictions):
        predictions[..., :4] = self.xywh2p1p2(predictions[..., :4])
        outputs = [None for _ in range(len(predictions))]

        for i, image_pred in enumerate(predictions):
            image_pred = image_pred[image_pred[:, 4] >= self.thresh]

            # If none anchor are remaining => process next image
            if not image_pred.size(0):
                continue

            # Object confidence times class confidence  (n, ) * (n, )
            score = image_pred[:, 4] * image_pred[:, 5:].max(1)[0]
            class_confs, class_preds = image_pred[:, 5:].max(1, keepdim=True)

            detections = torch.cat((image_pred[:, :5],
                                    class_confs.type(predictions.dtype),
                                    class_preds.type(predictions.dtype)), dim=1)

            keep = batched_nms(image_pred[:, :4].float(), score, class_preds[:, 0], self.nms_thresh)
            outputs[i] = detections[keep]

        return outputs

    @staticmethod
    def xywh2p1p2(x):
        y = x.new(x.shape)
        y[..., 0] = x[..., 0] - x[..., 2] / 2.
        y[..., 1] = x[..., 1] - x[..., 3] / 2.
        y[..., 2] = x[..., 0] + x[..., 2] / 2.
        y[..., 3] = x[..., 1] + x[..., 3] / 2.
        return y

    @staticmethod
    def p1p2Toxywh(x):
        y = x.new(x.shape)
        y[..., 0] = x[..., 0]
        y[..., 1] = x[..., 1]
        y[..., 2] = x[..., 2] - x[..., 0]
        y[..., 3] = x[..., 3] - x[..., 1]
        return y

    def _postprocess(self, outputs, shapes, box_format='ltrb', classes=None, buffer_ratio=0.0):
        outputs = self._nms(outputs)
        
        detections = []
        for i, frame_bbs in enumerate(outputs):
            im_height, im_width, _ = shapes[i]
            if frame_bbs is None:
                detections.append([])
                continue

            frame_bbs = self._resize_boxes(frame_bbs, self.model_image_size, (im_height, im_width))
            frame_dets = []
            for box in frame_bbs:
                pred_box = self.p1p2Toxywh(box[:4]).data.cpu().numpy()
                # box = box.data.cpu().numpy()
                cls_conf = box[4].item()
                cls_id = box[-1]
                cls_name = self.class_names[int(cls_id)]

                if classes is not None and cls_name not in classes:
                    continue

                left, top, w, h = pred_box
                right = left + w
                bottom = top + h
                
                width = right - left + 1
                height = bottom - top + 1
                width_buffer = width * buffer_ratio
                height_buffer = height * buffer_ratio

                top = max( 0.0, top-0.5*height_buffer )
                left = max( 0.0, left-0.5*width_buffer )
                bottom = min( im_height - 1.0, bottom + 0.5*height_buffer )
                right = min( im_width - 1.0, right + 0.5*width_buffer )

                box_infos = []
                for c in box_format:
                    if c == 't':
                        box_infos.append( int(round(top)) ) 
                    elif c == 'l':
                        box_infos.append( int(round(left)) )
                    elif c == 'b':
                        box_infos.append( int(round(bottom)) )
                    elif c == 'r':
                        box_infos.append( int(round(right)) )
                    elif c == 'w':
                        box_infos.append( int(round(width+width_buffer)) )
                    elif c == 'h':
                        box_infos.append( int(round(height+height_buffer)) )
                    else:
                        assert False,'box_format given in detect unrecognised!'
                assert len(box_infos) > 0 ,'box infos is blank'

                detection = (box_infos, cls_conf, cls_name)
                frame_dets.append(detection)
            detections.append(frame_dets)

        return detections

    @staticmethod
    def _resize_boxes(boxes, current_dim, original_shape):
        h_ratio = original_shape[0] / current_dim[0]
        w_ratio = original_shape[1] / current_dim[1]
        boxes[..., 0] *= w_ratio
        boxes[..., 1] *= h_ratio
        boxes[..., 2] *= w_ratio
        boxes[..., 3] *= h_ratio
        return boxes


if __name__ == '__main__':
    import cv2

    imgpath = 'test.jpg'

    device = 0
    yolov4 = YOLOV4( 
        score=0.5, 
        bgr=True, 
        # batch_size=num_vid_streams,
        # gpu_usage=od_gpu_usage, 
        gpu_device='cuda:{}'.format(device),
        model_image_size=(608, 608),
        max_batch_size = 4,
        half=True
    )
    img = cv2.imread(imgpath)
    bs = 5
    imgs = [ img for _ in range(bs) ]
    # img2 = cv2.resize(img, (200,200))

    n = 10
    dur = 0
    for _ in range(n):
        torch.cuda.synchronize()
        tic = perf_counter()
        # dets = yolov4.detect_get_box_in(imgs, box_format='ltrb', classes=None, buffer_ratio=0.0)[0]
        dets = yolov4.detect_get_box_in(imgs, box_format='ltrb', classes=['person'], buffer_ratio=0.0)[0]
        # print('detections: {}'.format(dets))
        torch.cuda.synchronize()
        toc = perf_counter()
        dur += toc - tic
    print('Average time taken: {:0.3f}s'.format(dur/n))

    cv2.namedWindow('', cv2.WINDOW_NORMAL)
    draw_frame = img.copy()
    for det in dets:
        # print(det)
        bb, score, class_ = det 
        l,t,r,b = bb
        cv2.rectangle(draw_frame, (l,t), (r,b), (255,255,0), 1 )
    
    cv2.imwrite('test_out.jpg', draw_frame)
    cv2.imshow('', draw_frame)
    cv2.waitKey(0)