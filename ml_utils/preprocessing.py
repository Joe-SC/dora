import pandas as pd
import numpy as  np

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.base import TransformerMixin

from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler


def decode(row_subset):
    if row_subset.sum() > 1:
        raise ValueError('Multiple Options Selected')
        
    for col in row_subset.index:
        if row_subset[col]:
            return col


def decode_feature(df, pattern, cname='', split=True, drop=True):
    ohe_cols = [col for col in df.columns if col.startswith(pattern)]
    if cname == '':
        cname = pattern[:pattern.rindex('.')]
    decoded = df[ohe_cols].apply(decode, axis=1)
    if split:
        decoded = (
            decoded
                .str.split('.')
                .str[-1]
        )        
    if drop:
        df = df.drop(labels=ohe_cols, axis=1)
        if len([col for col in df if col.startswith(pattern)]) != 0:
            raise Exception('Bad pattern supplied')
    
    df[cname] = decoded
    return df


def target_melt(df, value_vars, var_name='target', split=True):
    id_vars = [col for col in df if col not in value_vars]
    melted_df = pd.melt(df, id_vars=id_vars, value_vars=value_vars, var_name=var_name)
    if split:
        melted_df[var_name] = (
            melted_df[var_name]
                .str.split('.')
                .str[-1]
        )
    return (
        melted_df
            .loc[melted_df['value']]
            .drop(labels=['value'], axis=1)
            .reset_index(drop=True)
    )


class CategoricalEncoder(TransformerMixin):
	"""
	CategoricalEncoder

	Description
	-----------
		Performs label encoding of DataFrame columns

	Attributes
	----------
		cols (list):
			List of columns to categorically encode

		encoders (dict):
			Dictionary of label encodings

		unseen (int): -9999 by defail
			The integer value we wish to use to encode previously unseen categories


	"""
	def __init__(self, cols=None, nan_values = [], unseen=-9999):
		self.cols = cols
		self.encoders = None
		self.unseen = unseen
		self.fitted = False
		self.nan_values = nan_values

	def fit(self, X, y=None):
		"""
		Store categories and encodings in our training data

		Usage: Apply this to our training data only to learn encodings
		"""

		# If cols not specified, do for all DataFrame columns
		if self.cols:
			self.encoders = {col: {} for col in self.cols}
		else:
			self.encoders = {col: {} for col in X.columns.tolist()}

		for col, enc in self.encoders.items():
			# get unique values in the column to encode
			values = X[col].value_counts().index.to_list()
			# store unique values alongside label encoding
			self.encoders[col] = dict(zip(values, range(len(values))))
		self.fitted = True
		return self

	def transform(self, X):
		"""
		Encode categories as integers using stored encodings

		Usage: Apply to both training data and validation/test data to encode categoricals
		"""
		# encode a copy of the trianing data so we don not change it
		encoded_df = X.copy()
		if not self.fitted:
			raise Exception('Encoder not fitted yet')

		# Use our encoders to encode every categorical column
		for col, enc in self.encoders.items():

			# First make null values truly null so we can fillna()
			encoded_df.loc[encoded_df[col].isin(self.nan_values), col] = None

			# Do label encoding
			encoded_df[col] = (
				encoded_df[col]
					.map(lambda key: enc.get(key))
					.fillna(self.unseen)
					.astype(int)
				)
		return encoded_df



class ImputedPCA(TransformerMixin):
	"""
	Imputed PCA

	Description
	-----------
		Applies PCA with scaling and imputing based on values in training data
		The idea here is that we can combine highly correlated features into one
		using Principal Component Analysis


	Attributes
	----------
		n_components (int): default 1
			number of components to reduce to via PCA

		imp_strategy (str): default 'mean'
			imputation strategy for filling nulls

		fill_value: default None
			imputation strategy for imputation with a constant value
			by default use sklearn's SimpleImputer strategy

		scale strategy (str): default None
			scaling strategy to use
			by default apply no scaling
	"""
	def __init__(self, n_components=1, imp_strategy='mean', fill_value=None, scale_strategy=None):
		self.n_components = n_components
		if imp_strategy in ['mean', 'median',' most_frequent', 'constant']:
			if imp_strategy =='constant':
				self.imputer = SimpleImputer(strategy=imp_strategy, fill_value=fill_value)
			elif imp_strategy in ['mean', 'median',' most_frequent']:
				self.imputer = SimpleImputer(strategy=imp_strategy)
			else:
				raise ValueError('Invalid Imputation Strategy')

		if scale_strategy == 'standard':
			self.scaler = StandardScaler()
		elif scale_strategy == 'minmax':
			self.scaler = MinMaxScaler()
		elif scale_strategy == 'maxabs':
			self.scaler = MaxAbsScaler()
		else:
			self.scaler = None

		self.pca = PCA(n_components=n_components)


	def fit(self, X, y=None):
		X_ = self.imputer.fit_transform(X)
		if self.scaler is not None:
			X_ = self.scaler.fit_transform(X_)
		self.pca.fit(X_)

	def transform(self, X):
		X_ = self.imputer.transform(X)
		if self.scaler is not None:
			X_ = self.scaler.transform(X_)
		return self.pca.transform(X_)
