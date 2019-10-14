import numpy as np
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG,)
logger = logging.getLogger("__main__")

def __compute_multi_iou(boxA, boxB, d):
    """
    
    Parameters:
    -----------
    boxA : ndarray
    2*d, first d is x and rest is l
    """
    boxAarea = 1
    boxBarea = 1
    intersection = 1
    #IOU for hyperrectangles can be the product of multiple 1-dimensional lines
    for i in range(d):
        xa = max(boxA[i], boxB[i])
        #boxA[d+i] is the length for dimension i
        xb = min(boxA[i]+boxA[d+i], boxB[i]+boxB[d+i])
        
        intersection *= max(0, xb-xa)
        boxAarea *= boxA[d+i]
        boxBarea *= boxB[d+i]
    return intersection/ (boxAarea+boxBarea-intersection)

def min_dist(boxes, proposed):
    mins = []
    for p in proposed:
        mins.append(np.min(np.linalg.norm(np.array(boxes)-p, axis=1)))
    return np.mean(mins)

def compute_boxes(multi, dims):
    boxes = []
    if not multi:
        x_d = (np.ones(dims)*0.6).reshape(1,dims)
        l_d = (np.ones(dims)*0.3).reshape(1,dims)
        box0 = np.column_stack((x_d, l_d)).flatten()#np.array([0.6, 0.3])
        boxes.append(box0)
    else:
        x_d = (np.zeros(dims)).reshape(1,dims)
        l_d = (np.ones(dims)*0.2).reshape(1,dims)
        box0 = np.column_stack((x_d, l_d)).flatten()#np.array([0.6, 0.3]) 
        boxes.append(box0)        
        x_d = (np.ones(dims)*0.3).reshape(1,dims)
        l_d = (np.ones(dims)*0.2).reshape(1,dims)
        box1 = np.column_stack((x_d, l_d)).flatten()#np.array([0.6, 0.3]) 
        boxes.append(box1)        
        x_d = (np.ones(dims)*0.6).reshape(1,dims)
        l_d = (np.ones(dims)*0.2).reshape(1,dims)
        box2 = np.column_stack((x_d, l_d)).flatten()#np.array([0.6, 0.3])    
        boxes.append(box2)
    assert (len(boxes)==1 and not multi) or (len(boxes)==3 and multi )
    return boxes

def compute_iou(boxes, proposed, multi,dims):
    IOUs = []
    for gbox in boxes:
        for box in proposed:
            IOUs.append(__compute_multi_iou(gbox, box, dims))
    logger.debug('length of IOUs %d' % len(IOUs))
    if multi:
        maxi = np.max(np.array(IOUs).reshape((3,len(proposed))),axis=1)
        for i,mx_value in enumerate(maxi):
            logger.info('Maximum IOU for box %d is : %f' % (i+1, mx_value))
        iou_metric = np.mean(maxi)
    else:
        iou_metric = np.max(IOUs)
        logger.info('Maximum IOU of proposed boxes %f' %iou_metric)
    return iou_metric
    
def custom_objective(y_true, y_hat):
    Delta_hat = y_hat[1:]-y_hat[:-1]
    Delta = y_true[1:] - y_true[:-1]
    grad = 2*(Delta_hat-Delta)
    hess = np.ones((len(Delta_hat)+1))
    grad = np.insert(grad,0,Delta[0]-Delta_hat[0])
    return grad, hess
 