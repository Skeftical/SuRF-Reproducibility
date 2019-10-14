import numpy as np
import time

class GlowWorm():
    
    def _adapt_radius(self,rd, N):
        return min(self.rs, max(0, rd + self.beta* (self.nt - len(N)) ) )
    
    def _move(self,xi, xj):
        xj = xj.reshape(1,-1)
        xi = xi.reshape(1,-1)
        return xi + self.s*((xj-xi)/np.linalg.norm(xj-xi) )
    
    def _probability_of_moving(self,lj, li, sN):
        loss = lj-li
        return loss/sN
    
    def _return_neighbourhood(self,x, rd, l, i):
        ix = (np.linalg.norm(self.X-x,axis=1)<rd) & (l<self.L)
        ix[i] = False
        return np.nonzero(ix)[0]#returns tuple of arrays
    
    def _luciferin_update(self,l, x):
        return (1-self.rho)*l+self.gamma*self.opt_func(x.reshape(1,-1))
    
    def optimize(self):
        for k in range(self.iter_max):
        #Luciferin update phase
            if self.time_capture:
                start = time.time()
            for i in range(self.L.shape[0]):
                self.L[i]= self._luciferin_update(self.L[i],self.X[i])
            if self.time_capture:
                end = time.time() - start
                self.SUM_TIME += end
            #Prune away invalid glowworms
            if self.prune and k==0:
                valid = self.L!=-np.inf
                self.X = self.X[valid]
                self.R = self.R[valid]
                self.L = self.L[valid]
            #Movement phase
            for i in range(self.X.shape[0]):
                N = self._return_neighbourhood(self.X[i], self.R[i], self.L[i], i)
                max_ = None
                max_j = None
                if len(N) == 0:
                    #First adapt its radius so that it won't die out.
                    self.R[i] = self._adapt_radius(self.R[i], N)                    
#                     print('Worm {0} has no neighbours'.format(i))
                    continue;
                for j in N:
                    sN = np.sum(self.L[N]-self.L[i])
                    pij = self._probability_of_moving(self.L[j], self.L[i], sN)
                    if max_ is None or pij>max_:
                        max_j = j
                        max_ = pij
                self.X[i] = self._move(self.X[i], self.X[max_j])
                self.R[i] = self._adapt_radius(self.R[i], N)
            if self.trace:
                self.history = np.column_stack((self.history, self.X))
            
        return self.X
    
    def __init__(self,opt_func, dimensions=1, glowworms=50, s=0.03, r0=2, iter_max=100, 
                 rho=0.4,gamma=0.6,beta=0.08,nt=5, search_space=(0,1), log_trace=False, prune=False, time_capture=False):
        '''
        GlowWorm for multi-modal optimization
        args :
        ----------------------
        opt_func : callable
            optimization function, has to accept an input parameter
        dimensions : [int]
            Number of dimensions
        glowworms : [int]
            Number of particles
        s : float
            Step size
        r0 : float
            Initial neighbourhood size
        iter_max : int
            Maximum number of iterations
        rho : float
            Luciferin decay constant
        gamma : float
            Luciferin enhancement constant
        beta : float
            Neighbourhood discount
        search_space : tuple \in R^d
            Search space across all dimensions
        log_trace : boolean
            Logs the path of glowworms for debugging. The paths are then available through self.history.
            Feature is disabled for over 2-d
            
        '''
        DIMENSIONS = dimensions
        GLOWWORMS = glowworms
        self.opt_func = opt_func
        self.s = s
        self.r0 = r0
        self.X = np.zeros((GLOWWORMS,DIMENSIONS))
        random_positions = np.random.uniform(low=search_space[0], high=search_space[1], size=(GLOWWORMS,DIMENSIONS))
        self.X+=random_positions
        self.L = np.zeros(GLOWWORMS)+5
        self.R = np.ones(GLOWWORMS)*r0
        self.iter_max = iter_max
        #Constants
        self.rho = rho #lucifering decay
        self.gamma = gamma #lucifering enhancement
        self.beta = beta
        self.nt = nt
        self.rs = r0
        self.trace = False
        self.prune = prune
        self.time_capture = time_capture
        self.SUM_TIME = 0
        if log_trace==True:
            self.trace = True
            self.history = self.X

class GlowWormDensity():
    
    def _adapt_radius(self,rd, N):
        return min(self.rs, max(0, rd + self.beta* (self.nt - len(N)) ) )
    
    def _move(self,xi, xj):
        xj = xj.reshape(1,-1)
        xi = xi.reshape(1,-1)
        return xi + self.s*((xj-xi)/np.linalg.norm(xj-xi) )
    
    def _probability_of_moving(self,lj, li, sN):
        loss = lj-li
        return loss/sN
    
    def _return_neighbourhood(self,x, rd, l, i):
        ix = (np.linalg.norm(self.X-x,axis=1)<rd) & (l<self.L)
        ix[i] = False
        return np.nonzero(ix)[0]#returns tuple of arrays
    
    def _luciferin_update(self,l, x):
        return (1-self.rho)*l+self.gamma*self.opt_func(x.reshape(1,-1))
    
    def optimize(self):
        for k in range(self.iter_max):
        #Luciferin update phase
            for i in range(self.L.shape[0]):
                self.L[i]= self._luciferin_update(self.L[i],self.X[i])

            #Movement phase
            for i in range(self.X.shape[0]):
                N = self._return_neighbourhood(self.X[i], self.R[i], self.L[i], i)
                max_ = None
                max_j = None
                if len(N) == 0:
                    #First adapt its radius so that it won't die out.
                    self.R[i] = self._adapt_radius(self.R[i], N)
#                     print('Worm {0} has no neighbours'.format(i))
                    continue;
                for j in N:
                    sN = np.sum(self.L[N]-self.L[i])
                    if self.probx is not None:
                        pij = self._probability_of_moving(self.L[j], self.L[i], sN)*float(self.probx(self.X[j]))
                    else:
                        pij = self._probability_of_moving(self.L[j], self.L[i], sN)
                    if max_ is None or pij>max_:
                        max_j = j
                        max_ = pij
                self.X[i] = self._move(self.X[i], self.X[max_j])
                if self.trace:
                    self.history = np.column_stack((self.history, self.X))
                self.R[i] = self._adapt_radius(self.R[i], N)
        return self.X
    
    def __init__(self,opt_func, dimensions=1, glowworms=50, s=0.03, r0=2,iter_max=100, 
                 rho=0.4,gamma=0.6,beta=0.08,nt=5, search_space=(0,1), log_trace=False, probx=None):
        '''
        GlowWorm for multi-modal optimization
        args :
        ----------------------
        opt_func : callable
            optimization function, has to accept an input parameter
        dimensions : [int]
            Number of dimensions
        glowworms : [int]
            Number of particles
        s : float
            Step size
        r0 : float
            Initial neighbourhood size
        iter_max : int
            Maximum number of iterations
        rho : float
            Luciferin decay constant
        gamma : float
            Luciferin enhancement constant
        beta : float
            Neighbourhood discount
        search_space : tuple \in R^d
            Search space across all dimensions
        log_trace : boolean
            Logs the path of glowworms for debugging. The paths are then available through self.history.
            Feature is disabled for over 2-d
            
        '''
        DIMENSIONS = dimensions
        GLOWWORMS = glowworms
        self.opt_func = opt_func
        self.s = 0.03
        self.r0 = r0
        self.X = np.zeros((GLOWWORMS,DIMENSIONS))
        random_positions = np.random.uniform(low=search_space[0], high=search_space[1], size=(GLOWWORMS,DIMENSIONS))
        self.X+=random_positions
        self.L = np.zeros(GLOWWORMS)+5
        self.R = np.ones(GLOWWORMS)*r0
        self.iter_max = iter_max
        #Constants
        self.rho = 0.4 #lucifering decay
        self.gamma = 0.6 #lucifering enhancement
        self.beta = 0.08
        self.nt = 5
        self.rs = self.r0
        self.trace = False
        if dimensions==2 and log_trace==True:
            self.trace = True
            self.history = self.X
        self.probx = probx            