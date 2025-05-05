#!/usr/bin/env pyhton
# -*- coding: UTF-8 -*-


__author__ = 'Chao Wu'
__date__ = '5/31/2019'
__version__ = '1.0'




import os
import re
import numpy as np
import pandas as pd
from multiprocessing import Pool
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor
from output import display_selected_features, save_selected_features
		
		


def select_features_using_RF(Xtrain, Xtest, Ytrain, Ytest, threshold, maxFeatures):
	#使用随机森林回归模型选择重要特征
	'''
	Parameters
	Xtrain: df, feature matrix for training
	Xtest: df, feature matrix for testing
	Ytrain: ser, target for training
	Ytest: ser, target for testing
	threshold: 'mean' or 'median' or float, select features with importance above 'mean' or 'median' or threshold
	maxFeatures: int, max # of select features

	Returns
	XtrainNew: array, new feature matrix for training
	XtestNew: array, new feature matrix for testing
	feaImportance: ser, feature importance
	feaSelected: lst, selected features
	'''
	
	reg = RandomForestRegressor(n_estimators = 1000, random_state = 0, n_jobs = 3)
	
	reg.fit(Xtrain.values, Ytrain.values)
	
	feaImportance = pd.Series(reg.feature_importances_, index = Xtrain.columns)
	
	
	sfm = SelectFromModel(reg, threshold = threshold, max_features = maxFeatures, prefit = True)
	
	feaSelected = Xtrain.columns[sfm.get_support()].tolist()
	
	
	XtrainNew = sfm.transform(Xtrain)
	XtestNew = sfm.transform(Xtest)
	
	
	return XtrainNew, XtestNew, feaImportance, feaSelected


def features_selector(ratio, Xtrain, Xtest, Ytrain, Ytest, threshold, maxFeatures, outDir):
    # 对单个目标变量进行特征选择。
    '''
    Parameters
    ratios: str, ratio ID
    Xtrain: df, feature matrix for training
    Xtest: df, feature matrix for testing
    Ytrain: ser, target for training
    Ytest: ser, target for testing
    threshold: 'mean' or 'median' or float, select features with importance above 'mean' or 'median' or threshold
    maxFeatures: int, max # of select features
    outDir: str, output directory

    Returns
    XtrainNew: array, new feature matrix for training
    XtestNew: array, new feature matrix for testing
    '''

    print('\nratio ' + ratio)

    XtrainNew, XtestNew, feaImportance, feaSelected = select_features_using_RF(Xtrain, Xtest, Ytrain, Ytest, threshold, maxFeatures)

    subOutDir = r'%s/%s' % (outDir, ratio)
    os.makedirs(subOutDir, exist_ok=True)

    display_selected_features(feaImportance, feaSelected,feaSelected)
    save_selected_features(feaImportance, feaSelected, subOutDir)

    return XtrainNew, XtestNew


def select_features_in_parallel(ratios, Xtrain, Xtest, Ytrain, Ytest, criteria, outDir, njobs):
	#对多个目标变量进行特征选择。
	'''
	Parameters
	ratios: lst, ratio IDs
	Xtrain: df, feature matrix for training
	Xtest: df, feature matrix for testing
	Ytrain: df, targets for training
	Ytest: df, targets for testing
	outDir: str, output directory
	criteria: 'mean' or 'median' or float or int, selection criteria
	njobs: int, # of jobs to run in parallel

	Returns
	XtrainNew: array, new feature matrix for training
	XtestNew: array, new feature matrix for testing
	'''
	
	def parse_criteria(criteria, nfeatures):
		'''
		Parameters
		criteria: 'mean' or 'median' or float or int, selection criteria
		nfeatures: int, # of total features

		Returns
		threshold: 'mean' or 'median' or float, select features with importance above 'mean' or 'median' or threshold
		maxFeatures: int, max # of select features
		'''

		if criteria in ['mean', 'median']:
			threshold = criteria
			maxFeatures = None
			
		elif re.match(r'0\.\d+', criteria) or re.match(r'\d+[Ee][+-]?\d+', criteria):
			threshold = float(criteria)
			maxFeatures = None
			
		elif re.match(r'\d+\.?\d*%', criteria):
			threshold = -np.inf
			maxFeatures = int(float(criteria[:-1]) * 0.01 * nfeatures)
			
		elif criteria.isdigit():
			threshold = -np.inf
			maxFeatures = int(criteria)
			
		return threshold, maxFeatures
		
	
	threshold, maxFeatures = parse_criteria(criteria, Xtrain.columns.size)
	
	
	pool = Pool(processes = njobs)
	
	XdataNew = {}
	for ratio in ratios:
	
		res = pool.apply_async(func = features_selector, args = (ratio, Xtrain, Xtest, Ytrain[ratio], Ytest[ratio], threshold, maxFeatures, outDir))
	
		XdataNew[ratio] = res
	
	pool.close()	
	pool.join()
	
	for ratio, res in XdataNew.items(): XdataNew[ratio] = res.get()
	
	
	return XdataNew
	
	
	

		
		
		
		
		
		
		
		