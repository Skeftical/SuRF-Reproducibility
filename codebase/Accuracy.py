import numpy as np
import logging
from codebase.utils import *
from Optimization_Methods.GlowWorm import GlowWorm

class AccuracyRunnerApprox():
    YREQ = 1000
    C = 4
    ITER_MAX = 250
    SUGGESTIONS = 20
    
    def __objective(self,X):
        res = np.log(self.m.predict(X) - self.YREQ) - self.C *np.sum(np.log(1+X[:,X.shape[1]//2:]),axis=1)
        res[np.isnan(res)] = -np.inf
        return res
    
    def check_accuracy(self,proposed):
        #Construct boxes or box and check IOU
        #Multi-modal boxes regions are : [0,0.2]^d, [0.3,0.5]^d, [0.6,0.8]^d
        #Single regions : [0.6,0.9]^d
        boxes = compute_boxes(self.multi,self.dims)
        iou_metric = compute_iou(boxes, proposed, self.multi,self.dims)
        avg_min_dist = min_dist(boxes, proposed)
        logger.info('Average Minimum Distance {:.2f}'.format(avg_min_dist))
        logger.info('Finished Run =======================================')
        return iou_metric, avg_min_dist
    def get_j(self):
        return self.J
        
    def run_test(self):
        logger.info('Starting Optimization')
        gw = GlowWorm(self.__objective, dimensions=2*self.dims,glowworms=self.g,iter_max=self.ITER_MAX,r0=self.rad)
        gw.optimize()
        ix_sort = np.argsort(np.array(list(map(lambda x: float(-self.__objective(x.reshape(1,-1))),gw.X))).reshape(self.g,))
        self.J = list(filter(lambda x: x!=-np.inf, map(lambda x: float(self.__objective(x.reshape(1,-1))),gw.X)))
        proposed = gw.X[ix_sort,:][:self.SUGGESTIONS]
        assert proposed.shape == (min(self.SUGGESTIONS,50*(self.dims)),2*self.dims) or self.g is not None
        return proposed
    
    def scorer(self, estimator, X, y):
        self.m = estimator
        proposed = self.run_test()
        iou, avgmin = self.check_accuracy(proposed)
        return iou
    
    def __init__(self,dimensions, multiple_regions, type_of_aggregate, model, g=None, iters=None, suggestions='part'):
        self.dims = dimensions
        self.multi = multiple_regions
        self.aggr = type_of_aggregate
        
        if self.aggr=='aggr':
            self.YREQ =2
        self.m = model
        #Adapt number of glowworms and radius for dimensionality
        self.g = int(50*(self.dims)) if not g else g
        self.J = [0]*self.g
        if iters:
            self.ITER_MAX = iters
        self.rad = (1-0.5**(1/self.g))**(1/self.dims)
        if self.multi:
            self.SUGGESTIONS=self.g
            
        if suggestions=='all':
            self.SUGGESTIONS = self.g

   
        