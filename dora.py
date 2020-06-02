import pandas as pd
import numpy as np
import json
import seaborn as sns
import tqdm
import re
import pandas_profiling
from matplotlib import pyplot as plt
import seaborn as sns
import shap
from lime import lime_tabular
from ml_utils.preprocessing import target_melt
from ml_utils.tools import format_timedelta
import datetime

# from ml_utils.preprocessing import CategoricalEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from category_encoders import MEstimateEncoder, JamesSteinEncoder
from sklearn.preprocessing import MinMaxScaler
# from imblearn.combine import SMOTEEN
from imblearn.over_sampling import ADASYN
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, \
                            roc_curve, \
                            auc, \
                            classification_report, \
                            confusion_matrix, \
                            accuracy_score, \
                            precision_score, \
                            recall_score, \
                            f1_score

import dill

def read_dora(fname):
	with open(fname, 'rb') as f:
	    dora = dill.load(f)
	return dora


class Dora:
    def __init__(self, preprocessor, model, X, y, feature_names):
        self.preprocessor = preprocessor
        self.model = model
        self.X = X
        self.y = y
        self.feature_names = feature_names
        self.shap_explainer = shap.TreeExplainer(self.model)
        self.lime_explainer = lime_tabular.LimeTabularExplainer(X, training_labels=self.y, feature_names=feature_names, class_names=[False, True])
        self.shap_values = self.shap_explainer.shap_values(X)

    def __repr__(self):
    	return 'DORA: Data Oriented Risk Assessment'


    def shap_explain(self, raw_sample):
        processed_sample = self.preprocessor.transform(raw_sample.to_frame().transpose())
        sample_shap = self.shap_explainer.shap_values(processed_sample)
        shap.force_plot(self.shap_explainer.expected_value, sample_shap, processed_sample, link="logit")
        return (pd.Series(np.ravel(processed_sample), index=self.feature_names), sample_shap)


    def lime_explain(self, raw_sample):
        processed_sample = self.preprocessor.transform(raw_sample.to_frame().transpose())
        lime_exp = self.lime_explainer.explain_instance(processed_sample.ravel(), self.model.predict_proba, num_features=10, labels=(1,))
        lime_exp.show_in_notebook()
        return lime_exp


    def to_dora(self, fname):
    	with open(fname, 'wb') as f:
    		dill.dump(self, f)