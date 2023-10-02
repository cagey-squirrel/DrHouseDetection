# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Loss functions
"""

import torch
import torch.nn as nn

from utils.metrics import bbox_iou
from utils.torch_utils import de_parallel

import numpy as np
from matplotlib import pyplot as plt

def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    sort_obj_iou = False

    # Compute losses
    def __init__(self, model, autobalance=False, lf=None, conf_th=0, iou_th=0.8):
        self.loses_file = lf
        self.conf_rh = conf_th
        self.iou_th = iou_th
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device), reduction='none')
        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        m = de_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.na = m.na  # number of anchors
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.anchors = m.anchors
        self.device = device

        self.epoch = -1
        self.total_nc = 0
        self.total_iou = 0
        self.my_total_loss = torch.tensor([0])
        self.total_my_separate_losses = torch.tensor([0, 0 ,0])
        self.ord_total_loss = torch.tensor([0])
        self.total_ord_separate_losses = torch.tensor([0, 0 ,0])
        self.n = 0
        self.overlap = 0
        self.ord_minues_my = 0
        self.my_minus_ord = 0
        self.total_my_detections = 0
        self.real_num_iou = 0
        self.mse = nn.MSELoss(reduction='mean')

    def __call__(self, p, targets, index=1, epoch=2, mode='train'):
        if mode == "train":
            #total_ordinary_loss, separate_ordinary_losses, n, overlap, ord_minues_my, my_minus_ord, total_my_detections = self.ordinary_loss(p, targets)
            #return  total_ordinary_loss, separate_ordinary_losses
            
            # changed 2
            # my_total_loss, separate_my_losses, num_conf, num_iou = self.my_loss2(p, targets)
            my_total_loss, separate_my_losses, num_conf, num_iou, true_positives_list, real_num_iou = self.my_loss3(p, targets)
            total_ordinary_loss, separate_ordinary_losses, n, overlap, ord_minues_my, my_minus_ord, total_my_detections = self.ordinary_loss(p, targets, true_positives_list=true_positives_list)
            
            
            

            if epoch != self.epoch:

                my_loss_line = f'\nmy  loss = {self.my_total_loss}, separate my losses {self.total_my_separate_losses}'
                nc_niou_line = f'\nnc = {self.total_nc} niou = {self.total_iou} while targets = {self.n}, overlap = {self.overlap}, my_minus_ord = {self.my_minus_ord}, ord_minus_my = {self.ord_minues_my}, total_my_detections = {self.total_my_detections}, real_num_iou = {self.real_num_iou}'
                ord_loss_line = f'\n ord loss = {self.ord_total_loss}, separate ord losses {self.total_ord_separate_losses} \n'
                #print(my_loss_line)
                #print(ord_loss_line)
                self.loses_file.write(f'\nepoch: {self.epoch}')
                self.loses_file.write(my_loss_line)
                self.loses_file.write(nc_niou_line)
                self.loses_file.write(ord_loss_line)
                self.loses_file.flush()
                self.epoch = epoch

                self.total_nc = num_conf
                self.total_iou = num_iou
                self.my_total_loss = my_total_loss
                self.ord_total_loss = total_ordinary_loss
                self.total_my_separate_losses = separate_my_losses
                self.total_ord_separate_losses = separate_ordinary_losses
                self.n = n
                self.overlap = overlap
                self.ord_minues_my = ord_minues_my
                self.my_minus_ord = my_minus_ord
                self.total_my_detections = total_my_detections
                self.real_num_iou = real_num_iou
            else:
                self.total_nc += num_conf
                self.total_iou += num_iou
                self.my_total_loss += my_total_loss
                self.ord_total_loss += total_ordinary_loss
                self.total_my_separate_losses += separate_my_losses
                self.total_ord_separate_losses += separate_ordinary_losses
                self.n += n
                self.overlap += overlap
                self.ord_minues_my += ord_minues_my
                self.my_minus_ord += my_minus_ord
                self.total_my_detections += total_my_detections
                self.real_num_iou += real_num_iou

            loss = my_total_loss + total_ordinary_loss
            separate_losses = separate_my_losses + separate_ordinary_losses

            
            return loss, separate_losses

        elif mode == "valid":
            total_ordinary_loss, separate_ordinary_losses, n, overlap, ord_minues_my, my_minus_ord, total_my_detections = self.ordinary_loss(p, targets)
            return  total_ordinary_loss, separate_ordinary_losses
        else:
            raise Exception("Only train and valid modes are available, not {mode}")
        

    def ordinary_loss(self, p, targets, index=1, epoch=1, true_positives_list=[]):  # predictions, targets

        
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets
        overlap = 0
        ord_minues_my = 0
        my_minus_ord = 0
        total_targets = 0
        total_my_detections = 0
        layers_output = []
        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj

            

            n = b.shape[0]  # number of targets
            if n:
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # target-subset of predictions
                total_targets += n
                # Regression
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou  # iou ratio
                #tobj[b, a, gj, gi] = 1  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pcls, self.cn, device=self.device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(pcls, t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            #obji = self.BCEobj(pi[..., 4].sigmoid(), tobj)
            if len(true_positives_list):
                true_positives = true_positives_list[i]
                true_detections = (tobj > 0)
                no_detections = (tobj == 0)
                zeroed_detections = torch.zeros_like(tobj, device=self.device)
                positive_loss = (true_detections * self.BCEobj(pi[..., 4], tobj))#.sum() / (true_detections.sum() + 1)
                negative_loss = (no_detections * true_positives * self.BCEobj(pi[..., 4], zeroed_detections))#.sum() / (no_detections.sum() + 1)
                obji = (positive_loss + negative_loss).mean()

                my_detections = (~true_positives)
                my_detections_num = my_detections.sum()
                ord_detections_num = true_detections.sum()
                current_overlap = (true_detections * my_detections).sum()
                overlap += current_overlap
                total_my_detections += my_detections_num

                ord_minues_my += ord_detections_num - current_overlap
                my_minus_ord += my_detections_num - current_overlap
                #obji = self.BCEobj(pi[..., 4], tobj)
                #confs = pi[..., 4].sigmoid()
                #obji_matrix = self.BCEobj(pi[..., 4], tobj)
                #obji_matrix *= true_positives
                #obji = obji_matrix.mean()
            else:
                obji = self.BCEobj(pi[..., 4], tobj).mean()
            #obji = self.BCEobj(pi[..., 4], tobj)
            #if index == 0:
            #    #layers_output.append(pi[..., 4].detach().cpu().numpy().flatten())
            #    tobj_local = tobj.detach().cpu().numpy().flatten()
            #    layers_output.append(tobj_local)

            #pi_local = pi[..., 4].detach().cpu().numpy().flatten()
            #plt.hist(pi_local)
            #plt.savefig(f'histogram_in_pi_directly_epoch_{epoch}.png')
            #plt.clf()
            #plt.cla()
            #plt.close()
            
            total_layer_objectness_loss = obji * self.balance[i]  # obj loss
            lobj += total_layer_objectness_loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

            #print(f'this layer loss is {obji} and multiplier is {self.balance[i]} so total is {total_layer_objectness_loss}')
            #input(f'obj ord')

        #if index == 0:
        #    layers = np.concatenate((layers_output[0], layers_output[1], layers_output[2]))
        #    pred_confidences = layers.flatten()
        #    plt.hist(pred_confidences)
        #    plt.savefig(f'histogram_in_loss_epoch_{epoch}.png')
        #    plt.clf()
        #    plt.cla()
        #    plt.close()


        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= (self.hyp['box'])
        lobj *= self.hyp['obj']
        lcls *= (self.hyp['cls'])
        bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach(), total_targets, overlap, ord_minues_my, my_minus_ord, total_my_detections 


    def my_loss3(self, p, targets, stagod1=1, stagod2=2):

        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        
        num_conf = 0 # Broj celija koji imaju conf iznad th
        num_iou = 0
        real_num_iou = 0

        # Gain ima 7 jedinica
        gain = torch.ones(6, device=self.device)  # normalized to gridspace gain

        total_cells_accumulated = 0

        true_positives_list = []

        # Za svaki sloj 
        for i in range(self.nl):

            conf_loss = 0
            # Odrediti anchore za svaki layer i dimenzije grida za dati layer
            predictions = p[i]
            #object_present = torch.zeros(predictions[..., 4].shape, device=self.device, dtype=torch.bool)
            anchors, shape = self.anchors[i], predictions.shape
           
            # L1 shape = [16, 3, 160, 160, 11]
            # L2 shape = [16, 3, 80, 80, 11]
            # L3 shape = [16, 3, 40, 40, 11]

            # Gain ima svuda jedinice ali su mu 2 3 4 i 5 popunjeni brojem celija u datom sloju
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain
            

            # Match targets to anchors
            # Tehnicki, polja 2, 3, 4, 5 mnoze se brojem celija u datom redu 
            # Na mestu 4 i 5 su w i h, a na mestu 2 i 3 su centerX i centerY
            t = targets * gain  # shape(3,n,7)

            # Define
            # bc dobija ime slike i klasu, gxy su koordinate, gxw su dimenzije, a je anchor indeks
            bc, gxy, gwh = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors

            # a su i dalje anchori, b je ime slike a c klase 
            (b, c) = bc.long().T  #image, class


            gij = gxy.long()
            gi, gj = gij.T  # grid indices

            # Append
            # Postavis gj koordinate u opseg 0-width i 0-height


            # xy offseti i velicina detekcije
            tboxes = torch.cat((gxy - gij, gwh), 1)
            num_targets = targets.shape[0]

            # center_x, center_y, width, height, classes_prob

            pxy = predictions[..., 0:2]
            pxy = pxy.sigmoid() * 2 - 0.5
            
            pwh = predictions[..., 2:4]
            anchors = anchors[None, :, None, None, :]
            #print(f'\n\npwh dim = {pwh.shape} amnd anchors dim = {anchors.shape}\n\n')
            pwh = (pwh.sigmoid() * 2) ** 2 * anchors

            pboxes = torch.cat((pxy, pwh), -1)  # predicted box

            raw_conf = predictions[..., 4]
            conf = raw_conf.sigmoid()

            classes_logits = predictions[..., 5:]
            classes_probs = classes_logits.sigmoid()

            # c je klasa detekcije
            
            cells_to_try_indices = (conf > self.conf_rh)
            potential_objects_mask = torch.ones_like(cells_to_try_indices, device=self.device, dtype=torch.bool)

            if cells_to_try_indices.sum():

                num_conf += cells_to_try_indices.sum()
                # pboxes_to_try = pboxes[cells_to_try_indices]

                for target_index in range(num_targets):
                    

                    target_box = tboxes[target_index]
                    target_class = c[target_index]
                    target_image_index = b[target_index]

                    this_image_pboxes = pboxes[target_image_index]
                    this_image_cells = cells_to_try_indices[target_image_index]
                    pboxes_to_try = this_image_pboxes[this_image_cells]
                    #print(f'\n pboxes_to_try.shape {pboxes_to_try.shape}')
                    #print(f'\n cells_to_try_indices.shape {cells_to_try_indices.shape}')
                    #exit(-1)
                    
                    single_iou = bbox_iou(pboxes_to_try, target_box) #, CIoU=True)

                    # povecavanje na 0.85 usporava a sa 0.5 ne kovergira
                    true_positives = (single_iou > self.iou_th)
                    tp_sum = true_positives.sum()
                    if tp_sum == 0:
                        continue
                    potential_high_conf_objects = torch.ones_like(single_iou, device=self.device, dtype=torch.bool)
                    potential_high_conf_objects[true_positives] = 0

                    #num_of_zeros = (~potential_high_conf_objects).sum()
                    #sum_before = (potential_objects_mask[target_image_index, this_image_cells]).sum()
                    
                    #shape_left = potential_objects_mask[target_image_index, this_image_cells].shape
                    #print(f'shape left = {shape_left} and shape right = {potential_high_conf_objects.shape}')
                    potential_objects_mask[target_image_index, this_image_cells] *= potential_high_conf_objects.squeeze()

                    #sum_after = (potential_objects_mask[target_image_index, this_image_cells]).sum()
                    #if sum_after + num_of_zeros != sum_before:
                    #    print(f'\n VEC BILA NULA num sum_before = {sum_before}, sum_after = {sum_after}, num_of_zeros = {num_of_zeros} \n')

                    num_iou += tp_sum
                    #continue
                    true_positives = true_positives.flatten()
                    #tp_cells_num = true_positives.sum()
                
                    
                    ious = single_iou[true_positives]

                    #total_cells_accumulated += tp_cells_num
                    
                    #changed
                    #this_image_confs = raw_conf[target_image_index]
                    this_image_confs = conf[target_image_index]
                    confs_to_try = this_image_confs[this_image_cells]
                    confs_to_try_TP = confs_to_try[true_positives]
                    #print(f'confs_to_try_TP.shape = {confs_to_try_TP.shape}')
                    #print(f'ious.shape = {ious.shape}')
                    #print(f'tp_cells_num = {tp_cells_num}')
                    
                    ones_vector = torch.ones(confs_to_try_TP.shape, device=self.device)
                    #changed
                    #current_conf_loss = self.BCEobj(confs_to_try_TP, ious[:, 0])
                    #current_conf_loss = self.BCEobj(confs_to_try_TP, ones_vector)
                    current_conf_loss = self.mse(confs_to_try_TP, ones_vector)
                    current_conf_gain = 1.0 - current_conf_loss
                    
                    conf_loss += current_conf_gain

                    #print(f'\n current_conf_loss = {current_conf_loss}')
                    #input('confs')

                    # changed
                    #this_image_logits = classes_logits[target_image_index]
                    this_image_logits = classes_probs[target_image_index]
                    classes_to_try = this_image_logits[this_image_cells]
                    classes_to_try_TP = classes_to_try[true_positives]
                    #changed
                    #true_classes_vector = torch.full_like(classes_to_try_TP,  self.cn, device=self.device)
                    true_classes_vector = torch.full_like(classes_to_try_TP, 0, device=self.device)
                    #changed
                    #true_classes_vector[:, target_class] = self.cp
                    true_classes_vector[:, target_class] = 1
                    
                    #changed
                    #current_class_loss = self.BCEcls(classes_to_try_TP, true_classes_vector)
                    current_class_loss = self.mse(classes_to_try_TP, true_classes_vector)
                    current_class_gain = 1 - current_class_loss
                    #lcls += current_class_gain

                    #lbox += (1.0 - ious).mean()  
                    lbox += (ious).mean()  
            
            
            # Ovde im naplati IoU, class i obj ali napolju nemoj da im naplatis nista
            
            total_layer_conf_loss = self.balance[i] * conf_loss
            lobj += total_layer_conf_loss

            real_num_iou += (~potential_objects_mask).sum()
            #object_present[cells_to_try_indices] = potential_objects_mask
            true_positives_list.append(potential_objects_mask)
            #print(f'\n lobj is now{lobj} added tlcl {total_layer_conf_loss} where conf loss was {conf_loss} and multiplicator is {self.balance[i]}')
            #input('lobj')
        # promenjeno od total myloss 3 debugged

        lbox *= -self.hyp['box'] / 600
        lobj *= -self.hyp['obj'] / 10000
        lcls *= -self.hyp['cls'] / 1000

        #lbox *= -self.hyp['box'] / 1200
        #lobj *= -self.hyp['obj'] / 20000
        #lcls *= -self.hyp['cls'] / 2000
        bs = predictions.shape[0]  # batch size
        #print(f'\n total_acc = {total_cells_accumulated}')
        #print(f"\n lobj = {lobj} where multiplier is {5 * self.hyp['obj']} and loss was {lobj / (5 * self.hyp['obj'])}")
        
        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach(), num_conf, num_iou, true_positives_list, real_num_iou


    def my_loss2(self, p, targets, stagod1=1, stagod2=2):

        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        
        num_conf = 0 # Broj celija koji imaju conf iznad th
        num_iou = 0

        # Gain ima 7 jedinica
        gain = torch.ones(6, device=self.device)  # normalized to gridspace gain

        total_cells_accumulated = 0

        # Za svaki sloj 
        for i in range(self.nl):

            conf_loss = 0
            # Odrediti anchore za svaki layer i dimenzije grida za dati layer
            predictions = p[i]
            object_present = torch.zeros(predictions[..., 4].shape, device=self.device)
            anchors, shape = self.anchors[i], predictions.shape
           
            # L1 shape = [16, 3, 160, 160, 11]
            # L2 shape = [16, 3, 80, 80, 11]
            # L3 shape = [16, 3, 40, 40, 11]

            # Gain ima svuda jedinice ali su mu 2 3 4 i 5 popunjeni brojem celija u datom sloju
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain
            

            # Match targets to anchors
            # Tehnicki, polja 2, 3, 4, 5 mnoze se brojem celija u datom redu 
            # Na mestu 4 i 5 su w i h, a na mestu 2 i 3 su centerX i centerY
            t = targets * gain  # shape(3,n,7)

            # Define
            # bc dobija ime slike i klasu, gxy su koordinate, gxw su dimenzije, a je anchor indeks
            bc, gxy, gwh = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors

            # a su i dalje anchori, b je ime slike a c klase 
            (b, c) = bc.long().T  #image, class


            gij = gxy.long()
            gi, gj = gij.T  # grid indices

            # Append
            # Postavis gj koordinate u opseg 0-width i 0-height


            # xy offseti i velicina detekcije
            tboxes = torch.cat((gxy - gij, gwh), 1)
            num_targets = targets.shape[0]

            # center_x, center_y, width, height, classes_prob

            pxy = predictions[..., 0:2]
            pxy = pxy.sigmoid() * 2 - 0.5
            
            pwh = predictions[..., 2:4]
            anchors = anchors[None, :, None, None, :]
            #print(f'\n\npwh dim = {pwh.shape} amnd anchors dim = {anchors.shape}\n\n')
            pwh = (pwh.sigmoid() * 2) ** 2 * anchors

            pboxes = torch.cat((pxy, pwh), -1)  # predicted box

            raw_conf = predictions[..., 4]
            conf = raw_conf.sigmoid()

            classes_logits = predictions[..., 5:]
            classes_probs = classes_logits.sigmoid()

            # c je klasa detekcije
            
            cells_to_try_indices = (conf > 0.05)

            if cells_to_try_indices.sum():

                num_conf += cells_to_try_indices.sum()
                # pboxes_to_try = pboxes[cells_to_try_indices]

                for target_index in range(num_targets):
                    

                    target_box = tboxes[target_index]
                    target_class = c[target_index]
                    target_image_index = b[target_index]

                    this_image_pboxes = pboxes[target_image_index]
                    this_image_cells = cells_to_try_indices[target_image_index]
                    pboxes_to_try = this_image_pboxes[this_image_cells]
                    #print(f'\n pboxes_to_try.shape {pboxes_to_try.shape}')
                    #print(f'\n cells_to_try_indices.shape {cells_to_try_indices.shape}')
                    #exit(-1)
                    
                    single_iou = bbox_iou(pboxes_to_try, target_box, CIoU=True)

                    # povecavanje na 0.85 usporava a sa 0.5 ne kovergira
                    true_positives = (single_iou > 0.8).flatten()

                    tp_cells_num = true_positives.sum()
                    if tp_cells_num == 0:
                        continue
                
                    num_iou += tp_cells_num
                    ious = single_iou[true_positives]

                    total_cells_accumulated += tp_cells_num
                    
                    #changed
                    #this_image_confs = raw_conf[target_image_index]
                    this_image_confs = conf[target_image_index]
                    confs_to_try = this_image_confs[this_image_cells]
                    confs_to_try_TP = confs_to_try[true_positives]
                    #print(f'confs_to_try_TP.shape = {confs_to_try_TP.shape}')
                    #print(f'ious.shape = {ious.shape}')
                    #print(f'tp_cells_num = {tp_cells_num}')
                    
                    ones_vector = torch.ones(confs_to_try_TP.shape, device=self.device)
                    #changed
                    #current_conf_loss = self.BCEobj(confs_to_try_TP, ious[:, 0])
                    #current_conf_loss = self.BCEobj(confs_to_try_TP, ones_vector)
                    current_conf_loss = self.mse(confs_to_try_TP, ones_vector)
                    current_conf_gain = 1.0 - current_conf_loss
                    
                    conf_loss += current_conf_gain

                    #print(f'\n current_conf_loss = {current_conf_loss}')
                    #input('confs')

                    # changed
                    #this_image_logits = classes_logits[target_image_index]
                    this_image_logits = classes_probs[target_image_index]
                    classes_to_try = this_image_logits[this_image_cells]
                    classes_to_try_TP = classes_to_try[true_positives]
                    #changed
                    #true_classes_vector = torch.full_like(classes_to_try_TP,  self.cn, device=self.device)
                    true_classes_vector = torch.full_like(classes_to_try_TP, 0, device=self.device)
                    #changed
                    #true_classes_vector[:, target_class] = self.cp
                    true_classes_vector[:, target_class] = 1
                    
                    #changed
                    #current_class_loss = self.BCEcls(classes_to_try_TP, true_classes_vector)
                    current_class_loss = self.mse(classes_to_try_TP, true_classes_vector)
                    current_class_gain = 1 - current_class_loss
                    lcls += current_class_gain

                    #lbox += (1.0 - ious).mean()  
                    lbox += (ious).mean()  
            
            
            # Ovde im naplati IoU, class i obj ali napolju nemoj da im naplatis nista
            
            total_layer_conf_loss = self.balance[i] * conf_loss
            lobj += total_layer_conf_loss

            #print(f'\n lobj is now{lobj} added tlcl {total_layer_conf_loss} where conf loss was {conf_loss} and multiplicator is {self.balance[i]}')
            #input('lobj')
        lbox *= -self.hyp['box'] / 50
        lobj *= -self.hyp['obj'] / 2500
        lcls *= -self.hyp['cls'] / 100
        bs = predictions.shape[0]  # batch size
        #print(f'\n total_acc = {total_cells_accumulated}')
        #print(f"\n lobj = {lobj} where multiplier is {5 * self.hyp['obj']} and loss was {lobj / (5 * self.hyp['obj'])}")
        
        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach(), num_conf, num_iou


    def my_loss(self, p, targets, stagod1=1, stagod2=2):

        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        
        

        # Gain ima 7 jedinica
        gain = torch.ones(6, device=self.device)  # normalized to gridspace gain

        # Za svaki sloj 
        for i in range(self.nl):

            # Odrediti anchore za svaki layer i dimenzije grida za dati layer
            predictions = p[i]
            object_present = torch.zeros(predictions[..., 4].shape, device=self.device)
            anchors, shape = self.anchors[i], predictions.shape
           
            # L1 shape = [16, 3, 160, 160, 11]
            # L2 shape = [16, 3, 80, 80, 11]
            # L3 shape = [16, 3, 40, 40, 11]

            # Gain ima svuda jedinice ali su mu 2 3 4 i 5 popunjeni brojem celija u datom sloju
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain
            

            # Match targets to anchors
            # Tehnicki, polja 2, 3, 4, 5 mnoze se brojem celija u datom redu 
            # Na mestu 4 i 5 su w i h, a na mestu 2 i 3 su centerX i centerY
            t = targets * gain  # shape(3,n,7)

            # Define
            # bc dobija ime slike i klasu, gxy su koordinate, gxw su dimenzije, a je anchor indeks
            bc, gxy, gwh = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors

            # a su i dalje anchori, b je ime slike a c klase 
            (b, c) = bc.long().T  # anchors, image, class


            gij = gxy.long()
            gi, gj = gij.T  # grid indices

            # Append
            # Postavis gj koordinate u opseg 0-width i 0-height

            # Indeksi gde se nalaze centri detektovanih objekata po vrstama
            indices_width  = gj.clamp_(0, shape[2] - 1)
            # Indeksi gde se nalaze centri detektovanih objekata po kolonama
            indices_height = gi.clamp_(0, shape[3] - 1)
            # xy offseti i velicina detekcije
            tboxes = torch.cat((gxy - gij, gwh), 1)
            num_targets = targets.shape[0]

            # center_x, center_y, width, height, classes_prob

            pxy = predictions[..., 0:2]
            pxy = pxy.sigmoid() * 2 - 0.5
            
            pwh = predictions[..., 2:4]
            anchors = anchors[None, :, None, None, :]
            #print(f'\n\npwh dim = {pwh.shape} amnd anchors dim = {anchors.shape}\n\n')
            pwh = (pwh.sigmoid() * 2) ** 2 * anchors

            conf = predictions[..., 4]
            conf = conf.sigmoid()

            classes_logits = predictions[..., 5:]

            # c je klasa detekcije

            for image_index, image in enumerate(predictions):
                for anchor_index, anchor in enumerate(image):
                    for cell_index_x, cell_row in enumerate(anchor):
                        for cell_index_y, cell_values in enumerate(cell_row):
                            
                            cell_indices = image_index, anchor_index, cell_index_x, cell_index_y
                            if conf[cell_indices] < 0.2:
                                continue
                            
                            cell_px = pxy[cell_indices]
                            center_x, center_y = cell_px
                            cell_pwh = pwh[cell_indices]
                            pbox = torch.cat((cell_px, cell_pwh))  # predicted box

                            for target_index in range(num_targets):
                                target_tbox = tboxes[target_index]
                                target_index_x, target_index_y = indices_width[target_index], indices_height[target_index]
                                target_class = c[target_index]
                                iou = bbox_iou(pbox, target_tbox, CIoU=True)


                                if iou > 0.5 or (cell_index_x == target_index_x and cell_index_y == target_index_y):
                                    # Dodati loss za IoU i za klasifikaciju
                                    # Eventualno jos i dodati oznacavanje unutar matrice za racunanje conf lossa i za negative

                                    # print(f'IoU = {iou}, center_x={cell_index_x} center_x={cell_index_y} target_index_x={target_index_x} target_index_y={target_index_y}')
                                    iou_loss = (1.0 - iou)
                                    lbox += iou_loss
                                

                                    
                                    cell_class_probs = classes_logits[cell_indices]
                                    t = torch.full_like(cell_class_probs, self.cn, device=self.device)  # targets
                                    t[target_class] = self.cp
                                    lcls += self.BCEcls(cell_class_probs, t)  # BCE

                                    object_present[cell_indices] = 1
            
            conf_loss = self.BCEobj(predictions[..., 4], object_present)
            lobj += self.balance[i] * conf_loss

        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = predictions.shape[0]  # batch size

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()


    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)

        # Ovo je broj anchora i broj slika u ovom batchu
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        
        tcls, tbox, indices, anch = [], [], [], []

        # Gain ima 7 jedinica
        gain = torch.ones(7, device=self.device)  # normalized to gridspace gain

        # ai ima shape broj_anchora x broj_targeta
        # i to idu [[0,0,0,0,0], [1,1,1,1,1], [2,2,2,2,2]] za npr 3 anchora i 5 targeta
        # Dakle za svaki target ima po jedan indeks anchora
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)

        # Umnozi sve targete tako da su sada isti za svaki layer
        # Zatim na svakog od njih dodaj po indeks anchora
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # append anchor indices


        g = 0.5  # bias
        off = torch.tensor(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],  # j,k,l,m
                # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
            ],
            device=self.device).float() * g  # offsets

        # Za svaki sloj 
        for i in range(self.nl):


            predictions = p[i]
            # Odrediti anchore za svaki layer i dimenzije grida za dati layer
            anchors, shape = self.anchors[i], predictions.shape
           
            # L1 shape = [16, 3, 160, 160, 11]
            # L2 shape = [16, 3, 80, 80, 11]
            # L3 shape = [16, 3, 40, 40, 11]

            # Gain ima svuda jedinice ali su mu 2 3 4 i 5 popunjeni brojem celija u datom sloju
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain
            

            # Match targets to anchors
            # Tehnicki, polja 2, 3, 4, 5 mnoze se brojem celija u datom redu 
            # Na mestu 4 i 5 su w i h, a na mestu 2 i 3 su centerX i centerY
            t = targets * gain  # shape(3,n,7)
            if nt:  # Ako u ovom batchu ima oznaka
                # Matches

                # Odredi koji je odnos izmedju anchora i velicine targeta
                r = t[..., 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                # Ostavi samo anchore koji uvelicani do 4 puta mogu da prekriju ceo target
                t = t[j]  # filter

                
                #   t ovde ima shape [x, 7] gde je x neki broj
                # 7 vrednosti predstavljaju: indeks slike, 4 koordinate pravougaonika, indeks klase, indeks anchora
                
                
                 





                # Offsets
                # Gain 2, 3 je ono sto sadrzi ukupan broj celija
                # Posto je t[2,3] koji je u opsegu [0, 1] pomnozen brojem celija on je sada u opsegu [0, broj_celija]
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse

                # % 1 daje ostatak iza zareza i gledamo slucajeve kada je on manji od 0.5 i kada je veci od 0.5
                # Ali ne u pocetnim redovima (jer oni nemaju ulevo) pa je uslov da to budu brojevi veci od 1

                # j se odnosi na koordinate x, a k na koordinate y
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T


                ind = torch.stack((torch.ones_like(j), j, k, l, m))
                
                repeated_t = t.repeat((5, 1, 1))
                t = repeated_t[ind]

                
                # offsets kada se oduzmu od koordinata celija daju susedne celije, gornju, donju i ugao
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[ind]
            else:
                t = targets[0]
                offsets = 0

            # Define
            # bc dobija ime slike i klasu, gxy su koordinate, gxw su dimenzije, a je anchor indeks
            bc, gxy, gwh, a = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors

            # a su i dalje anchori, b je ime slike a c klase 
            a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class


            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid indices

            # Append
            # Postavis gj koordinate u opseg 0-width i 0-height
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid
            # Za box loss nam trebaju offseti od pocetka celije
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

            #print(f'\n anchors[a].shape = {anchors[a].shape}, indices ={indices[-1][0].shape}, tbox = {tbox[-1].shape}, tcls = {tcls[-1].shape}')
            #print(f'\n nt = {nt}')
            #print(f'\n targets.shape = {targets.shape}')
            #input('stop')

        return tcls, tbox, indices, anch
