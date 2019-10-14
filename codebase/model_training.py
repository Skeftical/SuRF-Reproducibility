import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from matplotlib import patches
from sklearn import metrics
import os
import sys
import itertools
from pathlib import Path
import pickle
import logging

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG,)
logger = logging.getLogger(__name__)

#model training
def train_and_score(queries,model,dims, **param_grid):
    X_train, X_test, y_train, y_test = train_test_split(
     queries[:,:queries.shape[1]-1], queries[:,-1], test_size=0.2, random_state=0)
    if param_grid:
        m = GridSearchCV(model(), cv=3,n_jobs=6,
                   param_grid= param_grid)
    else:
        m=model()
    
    m.fit(X_train, y_train)
    pred = m.predict(X_test)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, pred))
    r2 = metrics.r2_score(y_test, pred)
    print("RMSE : {}".format(rmse))
    print("Fitting : {}".format(r2))
    return m, rmse, r2


#Save models
xgb_parameters = {         "learning_rate": 10.0**-np.arange(1,4),
                           "max_depth" : np.arange(3,15,2),
                           "n_estimators": [100, 200, 300],
                           "reg_lambda": 10.0**-np.arange(0,4)
                 }
test_models = ['XGB']
directory = os.fsencode('../input/queries')

if __name__=="__main__":
    for file in os.listdir(directory):

        filename = os.fsdecode(file)
        if filename.startswith("queries"):
            a = filename.split('-')
            dims = int(a[2])
            f = np.loadtxt('../input/queries/%s' % (filename) ,delimiter=',')
            sample = f[np.random.randint(low=0, high=f.shape[0], size=5000)]

            logger.debug("File : {0}".format(filename))
            print(f.shape)
            assert f.shape[1]-1==2*dims
            for lm in test_models:
                if lm=='XGB':
                    m,RMSE,R2 = train_and_score(f,XGBRegressor, dims, **xgb_parameters)
            #         m = train_and_score(sample,GaussianProcessRegressor, dims)
                    #Store for later use
                    pkl_filename = "../models/{:}-XGB-RMSE={:.2f}-R2={:.2f}.pkl".format(filename,RMSE,R2)  
                    with open(pkl_filename, 'wb') as file:  
                        pickle.dump(m, file)         
