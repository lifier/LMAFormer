import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage import measure
import cv2
import numpy as np

class PD0_FA0():
    def __init__(self, nclass, thre):
        super(PD0_FA0, self).__init__()
        self.nclass = nclass
        self.thre = thre
        self.image_area_total = []
        self.image_area_match = []
        self.FA0 = 0
        self.PD0 = 0
        self.target = 0
        self.exlment_num = 0
    def update(self, preds, labels):

        predits = (preds > self.thre).astype('int64')
        labelss = labels.astype('int64')  # P

        image = measure.label(predits, connectivity=2)
        coord_image = measure.regionprops(image)
        label = measure.label(labelss, connectivity=2)
        coord_label = measure.regionprops(label)

        self.target += len(coord_label)
        self.image_area_total = []
        self.image_area_match = []
        self.distance_match = []
        self.dismatch = []

        for K in range(len(coord_image)):
            area_image = np.array(coord_image[K].area)
            self.image_area_total.append(area_image)
        label_PD = 0
        for i in range(len(coord_label)):
            centroid_label = np.array(list(coord_label[i].centroid))
            for m in range(len(coord_image)):

                centroid_image = np.array(list(coord_image[m].centroid))
                distance = np.linalg.norm(centroid_image - centroid_label)
                area_image = np.array(coord_image[m].area)
                if distance < 3:
                    self.distance_match.append(distance)
                    self.image_area_match.append(area_image)

                    del coord_image[m]
                    break
        self.dismatch = [x for x in self.image_area_total if x not in self.image_area_match]
        self.FA0 += np.sum(self.dismatch)
        self.PD0 += len(self.distance_match)
        self.exlment_num += predits.size

    def get(self):

        Final_FA0 = self.FA0 / self.exlment_num
        Final_PD0 = self.PD0 / self.target

        return Final_FA0, Final_PD0

    def reset(self):
        self.FA0 = 0
        self.PD0 = 0


class SigmoidMetric():
    def __init__(self):
        self.reset()
        self.IoU = []
    def update(self, pred, labels):
        correct, labeled = self.batch_pix_accuracy(pred, labels)
        iou = self.batch_intersection_union(pred, labels)

        self.total_correct += correct
        self.total_label += labeled
        self.IoU.append(iou)
    def get(self):
        """Gets the current evaluation result."""
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        temp_IoU = self.IoU
        iou = temp_IoU[-1]
        mIoU = sum(temp_IoU)/len(temp_IoU)
        return pixAcc, mIoU,iou

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self.total_inter = 0
        self.total_union = 0
        self.total_correct = 0
        self.total_label = 0

    def batch_pix_accuracy(self, output, target):
        assert output.shape == target.shape
        predict = (output > 0).astype('int64') # P
        pixel_labeled = np.sum(target > 0) # T
        pixel_correct = np.sum((predict == target)*(target > 0)) # TP
        assert pixel_correct <= pixel_labeled
        return pixel_correct, pixel_labeled

    def batch_intersection_union(self, output, target):
        target = target.astype(np.bool_)
        output = output.astype(np.bool_)

        if np.isclose(np.sum(target), 0) and np.isclose(np.sum(output), 0):
            return 1
        else:
            return np.sum((target & output)) / \
                np.sum((target | output), dtype=np.float32)


class SamplewiseSigmoidMetric():
    def __init__(self, nclass, score_thresh=0.5):
        self.nclass = nclass
        self.score_thresh = score_thresh
        self.reset()
        self.single_IoU = []
    def update(self, preds, labels):
        """Updates the internal evaluation result."""
        correct, labeled = self.batch_pix_accuracy(preds, labels)
        inter, union = self.batch_intersection_union(preds, labels)

        self.total_correct.append(correct)
        self.total_label.append(labeled)
        self.total_inter.append(inter)
        self.total_union.append(union)

    def get(self):
        """Gets the current evaluation result."""
        nIoU = np.sum(self.total_inter)/np.sum(self.total_union)
        return nIoU

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self.total_inter = []
        self.total_union = []
        self.total_correct = []
        self.total_label = []

    def batch_pix_accuracy(self, output, target):
        assert output.shape == target.shape
        predict = (output > 0).astype('int64') # P
        pixel_labeled = np.sum(target > 0) # T
        pixel_correct = np.sum((predict == target)*(target > 0)) # TP
        assert pixel_correct <= pixel_labeled
        return pixel_correct, pixel_labeled

    def batch_intersection_union(self, output, target):
        mini = 1
        maxi = 1  # nclass
        nbins = 1  # nclass
        predict = (output > 0).astype('int64')
        target = target.astype('int64')  # T
        intersection = predict * (predict == target)  # TP

        # areas of intersection and union
        area_inter, _ = np.histogram(intersection, bins=nbins, range=(mini, maxi))
        area_pred, _ = np.histogram(predict, bins=nbins, range=(mini, maxi))
        area_lab, _ = np.histogram(target, bins=nbins, range=(mini, maxi))
        area_union = area_pred + area_lab - area_inter
        assert (area_inter <= area_union).all()
        return area_inter, area_union


class ROCMetric():
    """Computes pixAcc and mIoU metric scores
    """

    def __init__(self, nclass, bins):
        super(ROCMetric, self).__init__()
        self.nclass = nclass
        self.bins = bins
        self.tp_arr = np.zeros(self.bins + 1)
        self.pos_arr = np.zeros(self.bins + 1)
        self.fp_arr = np.zeros(self.bins + 1)
        self.neg_arr = np.zeros(self.bins + 1)
        self.class_pos = np.zeros(self.bins + 1)
        # self.reset()

    def update(self, preds, labels):
        for iBin in range(self.bins + 1):
            score_thresh = (iBin + 0.0) / self.bins
            # print(iBin, '-th, score_thresh: ', score_thresh)
            i_tp, i_pos, i_fp, i_neg, i_class_pos = cal_tp_pos_fp_neg(preds, labels, self.nclass, score_thresh)
            self.tp_arr[iBin] += i_tp
            self.pos_arr[iBin] += i_pos
            self.fp_arr[iBin] += i_fp
            self.neg_arr[iBin] += i_neg
            self.class_pos[iBin] += i_class_pos

    def get(self):
        tp_rates = self.tp_arr / (self.pos_arr + 0.001)
        fp_rates = self.fp_arr / (self.neg_arr + 0.001)

        recall = self.tp_arr / (self.pos_arr + 0.001)
        precision = self.tp_arr / (self.class_pos + 0.001)

        return tp_rates, fp_rates, recall, precision

    def reset(self):
        self.tp_arr = np.zeros([11])
        self.pos_arr = np.zeros([11])
        self.fp_arr = np.zeros([11])
        self.neg_arr = np.zeros([11])
        self.class_pos = np.zeros([11])

class PD_FA():
    def __init__(self, nclass, bins):
        super(PD_FA, self).__init__()
        self.nclass = nclass
        self.bins = bins
        self.image_area_total = []
        self.image_area_match = []
        self.FA = np.zeros(self.bins + 1)
        self.PD = np.zeros(self.bins + 1)
        self.target = np.zeros(self.bins + 1)
        self.exlment_num = 0

    def update(self, preds, labels):

        for iBin in range(self.bins + 1):
            score_thresh = (iBin * ((preds.max() - preds.min()) / self.bins)
                            + preds.min())
            predits = (preds > score_thresh).astype('int64')
            labelss = labels.astype('int64')  # P

            image = measure.label(predits, connectivity=2)
            coord_image = measure.regionprops(image)
            label = measure.label(labelss, connectivity=2)
            coord_label = measure.regionprops(label)

            self.target[iBin] += len(coord_label)
            self.image_area_total = []
            self.image_area_match = []
            self.distance_match = []
            self.dismatch = []

            for K in range(len(coord_image)):
                area_image = np.array(coord_image[K].area)
                self.image_area_total.append(area_image)
            label_PD = 0
            for i in range(len(coord_label)):
                centroid_label = np.array(list(coord_label[i].centroid))
                connectivity = 8
                ret_label, thresholded_label, stats_label, centroids_label = cv2.connectedComponentsWithStats((labelss).astype('int8'), connectivity=connectivity)
                num_label_region = (thresholded_label == i+1)
                label_bool = num_label_region.astype(bool)
                predits_bool = predits.astype(bool)
                intersection = np.logical_and(label_bool.reshape(-1), predits_bool.reshape(-1))
                if np.any(intersection):
                     temp_PD_num = 1
                else:
                     temp_PD_num = 0
                label_PD += temp_PD_num
            false_num_pic = predits - labelss  # 预测图-标签%
            false_num_pic_true = (false_num_pic > 0).astype('int64')
            self.FA[iBin] += false_num_pic_true.sum()
            self.PD[iBin] += label_PD
            self.exlment_num += predits.size
    def get(self):

        Final_FA = self.FA / self.exlment_num
        Final_PD = self.PD / self.target

        return Final_FA, Final_PD

    def reset(self):
        self.FA = np.zeros([self.bins + 1])
        self.PD = np.zeros([self.bins + 1])


def cal_tp_pos_fp_neg(output, target, nclass, score_thresh):
    output = torch.from_numpy(output).unsqueeze(0).unsqueeze(1)
    target = torch.from_numpy(target).unsqueeze(0).unsqueeze(1)
    predict = (output > score_thresh).float()
    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError('Unknown target dimension')

    intersection = predict * ((predict == target).float())

    tp = intersection.sum()
    fp = (predict * ((predict != target).float())).sum()
    tn = ((1 - predict) * ((predict == target).float())).sum()
    fn = (((predict != target).float()) * (1 - predict)).sum()
    pos = tp + fn
    neg = fp + tn
    class_pos = tp + fp

    return tp, pos, fp, neg, class_pos
