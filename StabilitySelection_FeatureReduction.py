# Use a conda env and spyder to run stability selection

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
from stability_selection import StabilitySelection, plot_stability_path
import pandas as pd
# import pickle
import sklearn
from sklearn.metrics import roc_curve
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt


if __name__ == '__main__':
    
    data = pd.read_csv('../data/processed/dataBalanced.csv')
    
    data = data[(data['Label'] == 'C') | (data['Label'] == 'H0')]
    
    print(data.shape)

    y = list(data.pop('Label').values)
    X = data.values

    # print(y)
    
    # print(data.head())

    # base_estimator = Pipeline([
    #     ('scaler', StandardScaler()),
    #     ('model', LogisticRegression(penalty='l1'))
    # ])
    
    pipe = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(solver='lbfgs', verbose=0, n_jobs=1, random_state=42))
    ])  
    
    # selector = StabilitySelection(base_estimator=base_estimator, lambda_name='model__C',
    #                               lambda_grid=np.logspace(-5, -1, 50))
    
    Cs = np.logspace(-5, 5, 21)
    selector = StabilitySelection(base_estimator=pipe, 
                            lambda_name='clf__C', lambda_grid=Cs)
    

    selector.fit(X, y)


    selected_variables = selector.get_support(indices=True)
    selected_scores = selector.stability_scores_.max(axis=1)


    f = open('../data/processed/selected_variables.csv', 'w')
    f.write('feature' + ',' + 'score' + '\n')
    
    for n in range(len(selected_variables)):
        f.write(data.columns[selected_variables[n]] + ',' + str(selected_scores[n]) + '\n')

    f.close()
    
    