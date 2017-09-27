import os, math
import numpy as np
import pandas as pd
import seaborn as sns
import helpers

import time
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd

from sklearn import svm
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif, chi2, f_classif
from sklearn.metrics import log_loss

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV
from sklearn import *
import sklearn

class GeneVar():
	def __init__(self):
		print("gene_var model")

	def get_feature_list(self, data_full):
		"""get variation features"""
		data_full = data_full.drop(['Class'], axis=1)
		original_list = list(data_full.columns)
		drop_list = ['ID', 'Text', 'describe', 'location', 'alias_name','gene_family', 'locus_group', 'locus_type', 'name', 'prev_name']
		feature_list = [i for i in original_list if i not in drop_list]
		return feature_list

	def _preprocess(self, data):
		"""preprocess steps for the original data"""
		import copy
		import re
		from sklearn import preprocessing
		
		data_full = copy.deepcopy(data)
		for c in data_full.columns:
			if data_full[c].dtype == 'object':
				if c=='location':
					data_full[c+'_1'] = data_full[c].apply(lambda x: re.findall(r"[0-9X]+", x)[0])
					data_full[c+'_2'] = data_full[c].apply(lambda x: re.findall(r"[0-9X]+", x)[1])
					data_full[c+'_3'] = data_full[c].apply(lambda x: re.findall(r"[0-9X]+", x)[2] if len(re.findall(r"[0-9X]+", x))>2 else '0')
				if c in ['alias_name', 'gene_family', 'prev_name']:
					data_full[c][data_full[c].notnull()] = data_full[c][data_full[c].notnull()].apply(lambda x:' '.join(x))
					data_full[c][data_full[c].isnull()] = 'nan'
		for c in data_full.columns:
			if c in ['Gene','Variation','amino1','amino2','symbol','location_1','location_2','location_3']:
				lbl = preprocessing.LabelEncoder()
				data_full[c] = lbl.fit_transform(data_full[c].values)
		data_full['describe'] = data_full['alias_name'].str.cat(\
		                        [data_full['gene_family'],data_full['locus_group'], data_full['locus_type'],
		                         data_full['name'], data_full['prev_name']], sep=' ')
		return data_full

	def fit_feature(self, data_full):
		"""to fit the feature matrices on training"""
		data_full = self._preprocess(data_full)
		print(data_full.columns)
		print(data_full.shape)

		fu = pipeline.FeatureUnion(
			n_jobs = -1,
			transformer_list = [
		('standard', cust_regression_vals2()),
		('pi', pipeline.Pipeline([('describe', cust_txt_col('describe')), \
		                          ('tfidf', feature_extraction.text.TfidfVectorizer(ngram_range=(1, 3))), \
		                          ('mi', feature_selection.SelectKBest(mutual_info_classif, k=50))]))        
		])

		#normalizer = preprocessing.StandardScaler()

		pl = pipeline.Pipeline([('tokenize', fu)])#, ('normalize', normalizer)])

		pl.fit(data_full.drop(['Class'], axis=1), data_full['Class'])
		return pl

	def get_feature(self, pl, data_full):
		"""to get the feature matrices"""
		data_full = self._preprocess(data_full)
		if 'Class' in data_full.columns:
			return pl.transform(data_full.drop(['Class'], axis=1))
		else:
			return pl.transform(data_full)


class TfidfSvm():
	"""the model class for tf-idf trained with svm"""
	def __init__(self):
		print("this is a tfidf+svm model")

	def fit_feature(self, data_full):
		tfidf = TfidfVectorizer(
			min_df=1, max_features=16000, strip_accents='unicode',lowercase =True,
			analyzer='word', use_idf=True, 
			smooth_idf=True, sublinear_tf=True, stop_words = 'english')
		ffilter = SelectKBest(mutual_info_classif, k=100)

		pl = pipeline.Pipeline([('tfidf', tfidf), ('mi', ffilter)])

		pl.fit(data_full["Text"], data_full["Class"]-1)
		return pl

	def get_feature(self, pl, data_full):
			return pl.transform(data_full["Text"])


	def fit(self, data_full):
		"""takes as input the full original data matrix"""
		tfidf = TfidfVectorizer(
			min_df=1, max_features=16000, strip_accents='unicode',lowercase =True,
			analyzer='word', use_idf=True, 
			smooth_idf=True, sublinear_tf=True, stop_words = 'english')
		ffilter = SelectKBest(mutual_info_classif, k=3000)

		parameters = {
		"estimator__C": [10],
		"estimator__kernel": ['linear']
		#"estimator__degree": [2, 3]
		}

		clf = GridSearchCV(OneVsRestClassifier(svm.SVC(probability=True, class_weight='balanced')),\
					 param_grid=parameters, scoring='neg_log_loss', n_jobs=-1)

		pl = pipeline.Pipeline([('tfidf', tfidf), ('mi', ffilter), ('svm', clf)])

		pl.fit(data_full["Text"], data_full["Class"]-1)
		return pl

	def predict_proba(self, pl, data_full):
		"""takes as input the trained model and the data matrix that needs to be tested"""
		y_prob = pl.predict_proba(data_full["Text"])
		
		if 'Class' in data_full.columns:
			print("logloss:")
			print(log_loss(data_full["Class"]-1, y_prob, eps=1e-15, normalize=True, labels=range(9)))

		return y_prob


class GeneSvm():
	"""the model class for gene&variations trained with svm"""
	def __init__(self):
		print("this is a model based on only gene&variations")

	def _preprocess(self, data):
		"""preprocess steps for the original data"""
		import copy
		data_full = copy.deepcopy(data)
		data_full['Gene_Share'] = data_full.apply(lambda r: \
									sum([1 for w in r['Gene'].split(' ') if w in r['Text'].split(' ')]), axis=1)
		data_full['Variation_Share'] = data_full.apply(lambda r: \
									sum([1 for w in r['Variation'].split(' ') if w in r['Text'].split(' ')]), axis=1)
		for i in range(10):
			data_full['Gene_'+str(i)] = data_full['Gene'].map(lambda x: str(x[i]) if len(x)>i else '')
			data_full['Variation'+str(i)] = data_full['Variation'].map(lambda x: str(x[i]) if len(x)>i else '')

		for c in data_full.columns:
			if data_full[c].dtype == 'object':
				if c in ['Gene','Variation']:
					lbl = preprocessing.LabelEncoder()
					data_full[c+'_lbl_enc'] = lbl.fit_transform(data_full[c].values)  
					data_full[c+'_len'] = data_full[c].map(lambda x: len(str(x)))
					data_full[c+'_words'] = data_full[c].map(lambda x: len(str(x).split(' ')))
				elif c != 'Text':
					lbl = preprocessing.LabelEncoder()
					data_full[c] = lbl.fit_transform(data_full[c].values)
				if c=='Text': 
					data_full[c+'_len'] = data_full[c].map(lambda x: len(str(x)))
					data_full[c+'_words'] = data_full[c].map(lambda x: len(str(x).split(' '))) 

		return data_full

	def fit_feature(self, data_full):
		"""to fit the feature matrices on training"""
		data_full = self._preprocess(data_full)

		fu = pipeline.FeatureUnion(
				n_jobs = -1,
				transformer_list = [
			('standard', cust_regression_vals()),
			('pi1', pipeline.Pipeline([('Gene', cust_txt_col('Gene')), \
				('count_Gene', feature_extraction.text.CountVectorizer(analyzer=u'char', ngram_range=(1, 8))), \
				('tsvd1', decomposition.TruncatedSVD(n_components=20, n_iter=25, random_state=12))])),
			('pi2', pipeline.Pipeline([('Variation', cust_txt_col('Variation')), \
				('count_Variation', feature_extraction.text.CountVectorizer(analyzer=u'char', ngram_range=(1, 8))), \
				('tsvd2', decomposition.TruncatedSVD(n_components=20, n_iter=25, random_state=12))])),     
			])
		
		normalizer = preprocessing.StandardScaler()

		pl = pipeline.Pipeline([('tokenize', fu), ('normalize', normalizer)])

		return pl.fit(data_full.drop(['Class'], axis=1))


	def get_feature(self, pl, data_full):
		"""to get the feature matrices"""
		data_full = self._preprocess(data_full)
		if 'Class' in data_full.columns:
			return pl.transform(data_full.drop(['Class'], axis=1))	
		else:
			return pl.transform(data_full)


	def fit(self, data_full):
		"""takes as input the full original data matrix"""
		data_full = self._preprocess(data_full)

		fu = pipeline.FeatureUnion(
				n_jobs = -1,
				transformer_list = [
			('standard', cust_regression_vals()),
			('pi1', pipeline.Pipeline([('Gene', cust_txt_col('Gene')), \
				('count_Gene', feature_extraction.text.CountVectorizer(analyzer=u'char', ngram_range=(1, 8))), \
				('tsvd1', decomposition.TruncatedSVD(n_components=20, n_iter=25, random_state=12))])),
			('pi2', pipeline.Pipeline([('Variation', cust_txt_col('Variation')), \
				('count_Variation', feature_extraction.text.CountVectorizer(analyzer=u'char', ngram_range=(1, 8))), \
				('tsvd2', decomposition.TruncatedSVD(n_components=20, n_iter=25, random_state=12))])),     
			])
		

		normalizer = preprocessing.StandardScaler()

		parameters = {
		"estimator__C": [1],
		"estimator__kernel": ['rbf'],
		#"estimator__degree": [2, 3]
		}

		clf = GridSearchCV(OneVsRestClassifier(svm.SVC(probability=True, class_weight='balanced')), param_grid=parameters, scoring='neg_log_loss', n_jobs=-1)

		pl = pipeline.Pipeline([('tokenize', fu), ('normalize', normalizer), ('svm', clf)])

		pl.fit(data_full.drop(['Class'], axis=1), data_full['Class']-1)
		return pl

	def predict_proba(self, pl, data_full):
		data_full = self._preprocess(data_full)
		if 'Class' in data_full.columns:
			y_prob = pl.predict_proba(data_full.drop(['Class'], axis=1))
			print("logloss:")
			print(log_loss(data_full["Class"]-1, y_prob, eps=1e-15, normalize=True, labels=range(9)))
		else:
			y_prob = pl.predict_proba(data_full)
		return y_prob



class cust_regression_vals(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
	def fit(self, x, y=None):
		return self
	def transform(self, x):
		x = x.drop(['Gene', 'Variation','ID','Text'],axis=1).values
		return x

class cust_txt_col(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
	def __init__(self, key):
		self.key = key
	def fit(self, x, y=None):
		return self
	def transform(self, x):
		#print('hey')
		#print(x[self.key].apply(str))
		return x[self.key].apply(str)

class cust_regression_vals2(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
	def fit(self, x, y=None):
		return self
	def transform(self, x):
		x = x.drop(['ID', 'Text', 'describe', 'location', 'alias_name','gene_family', 'locus_group', 'locus_type', 'name', 'prev_name'], axis=1).values
		return x


