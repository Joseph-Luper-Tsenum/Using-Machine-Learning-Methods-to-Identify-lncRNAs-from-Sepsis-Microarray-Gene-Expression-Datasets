import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing


def applySmote(features, target, label, file):
    # Start SMOTE
    oversample = SMOTE()

    featuresSmote, targetSmote = oversample.fit_resample(features.values, target.values)

    featuresDfSmote = pd.DataFrame(data=featuresSmote, columns=features.columns)
    targetDfSmote = pd.DataFrame(data=targetSmote, columns=[label])

    featuresDfTotal = pd.concat([featuresDfSmote, targetDfSmote], axis=1)

    featuresDfTotal.to_csv(file, index = False)

    print(Counter(targetSmote))


# Top Features
imbalancedDataAllFeatures = pd.read_csv('../data/raw/exprs_ML_Complete_transposed.csv').set_index('PAT_ID')

targetDf = imbalancedDataAllFeatures.pop('Label')
featuresDf = imbalancedDataAllFeatures


# Apply Smote
applySmote(featuresDf, targetDf, 'Label', '../data/processed/dataBalanced.csv')
