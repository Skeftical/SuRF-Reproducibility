import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn import metrics
import os
import sys
import itertools
from pathlib import Path
import pickle
import logging
import time

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
                           "max_depth" : np.arange(3,10,2),
                           "n_estimators": [100, 200, 300],
                           "reg_lambda": 10.0**-np.arange(0,4)
                 }

file = os.fsencode('input/queries/queries-uniform-5-multi_True-density')




filename = os.fsdecode(file)

f = np.loadtxt(filename ,delimiter=',')
cols = [1,2,3]
print(f.shape)
queries_num = np.linspace(10000, f.shape[0],num=10).astype(int)
print(queries_num)
training_overhead = {}
training_overhead['time'] = []
training_overhead['queries'] = []
training_overhead['hypertuning'] = []
for no in queries_num:
    X = f[:no,:]
    logger.debug("File : {0}".format(X.shape[0]))

    start = time.time()
    m,RMSE,R2 = train_and_score(X,XGBRegressor, X.shape[1], **xgb_parameters)
    end = time.time()-start
    training_overhead['time'].append(end)
    training_overhead['queries'].append(no)
    training_overhead['hypertuning'].append(True)
    for i in range(5):
        start = time.time()
        m,RMSE,R2 = train_and_score(X,XGBRegressor, X.shape[1])
        end = time.time()-start
        training_overhead['time'].append(end)
        training_overhead['queries'].append(no)
        training_overhead['hypertuning'].append(False)

df = pd.DataFrame(training_overhead)
df.to_csv('output/training.csv')    
