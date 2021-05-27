"""
NYU DSGA 1004: Big Data, Final Project
Population bias model
Date: April, 2021

Usage:
    python MSD_BiasOnly.py /scratch/work/courses/DSGA1004-2021/MSD   out/msd_extension_baseline_models.pkl

"""
# import numpy as np
import pandas as pd
import os
import sys
import time
from lenskit import batch, topn, util
from lenskit.algorithms.bias import Bias
from lenskit.algorithms import Recommender
import pickle

def load_data(dirname, sample_frac = 1):
    ## final model reads the full set at sample_frac = 1
    train_set = pd.read_parquet(os.path.join(dirname, 'cf_train.parquet'))
    validation_set = pd.read_parquet(os.path.join(dirname, 'cf_validation.parquet'))
    test_set = pd.read_parquet(os.path.join(dirname, 'cf_test.parquet'))
    # sampling
    train_set = train_set.sample(frac= sample_frac)
    validation_set = validation_set.sample(frac = sample_frac)
    test_set = test_set.sample(frac = sample_frac)
    # print(validation_set.head(10))
    return train_set, validation_set, test_set

def eval(aname, algo, train, test, n = 500):
    fittable = util.clone(algo)
    fittable = Recommender.adapt(fittable)
    fittable.fit(train)
    users = test.user.unique()
    # now we run the recommender
    recs = batch.recommend(fittable, users, n)
    recs['Algorithm'] = aname
    return recs

def MSD_ext_baseline_models(train_set, validation_set, test_set,  damping, n = 500):
    bias = Bias(damping=damping)
    ## train_set requires user, items and rating columns: change column names to meet input requirements
    train_set = train_set.rename(columns={"user_id": "user", "track_id": "item", "count": "rating"})
    test_set = test_set.rename(columns={"user_id": "user", "track_id": "item", "count": "rating"})
    ## run bias baseline model and apply on test set, generate top 500 items for each user
    rec_test = eval("bias", bias, train_set, test_set, n)
    # evaluations: precision at k
    rla = topn.RecListAnalysis()
    rla.add_metric(topn.precision)
    rla.add_metric(topn.ndcg)
    test_result = rla.compute(rec_test, test_set)
    precision_k = test_result["precision"].mean()
    ndcg = test_result["ndcg"].mean()
    return precision_k, ndcg

if __name__ == "__main__":
    dirname = sys.argv[1]
    ## load data
    t0 = time.time()
    train_set, validation_set, test_set = load_data( dirname)
    load_time = (time.time()-t0)
    print('loading time: %.3f' % load_time)

    ## train model
    t1 = time.time()
    precision_k, ndcg = MSD_ext_baseline_models(train_set, validation_set, test_set, damping=5)
    train_time = (time.time() - t1)
    print('training time: %.3f' % train_time)
    print(f"precision at k: {precision_k}")
    print(f"ndcg at k: {ndcg}")
    results = {'model': "popularity-based baseline model", 'precision_k': precision_k, \
               "ndcg": ndcg, 'load_time':load_time, 'train_time': train_time}
    outdir = sys.argv[2]
    with open(outdir, "wb") as outfile:
        pickle.dump(results, outfile)