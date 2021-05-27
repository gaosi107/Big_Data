import os
import sys
import numpy as np
import itertools
from pyspark.sql import SparkSession
from pyspark.sql.functions import expr
import pyspark.sql.functions as F
from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.mllib.evaluation import RankingMetrics


def transformIndex(spark, index_column, data_pq):
    '''
    Funtionality: Transform a string index into numeric labelled index to accelerate
    :param spark: spark session
    :param index_column: column to be indexed
    :param data_pq: the data parquet
    :return: data parquet with transformed indices
    '''
    indexer = StringIndexer(inputCol=index_column, outputCol=index_column + "_indexed", handleInvalid='skip')
    indexed_data_pq = indexer.fit(data_pq).transform(data_pq)

    return indexed_data_pq


def downsample(spark, train_pq, val_pq, sample_size=0.01):
    '''
    Funtionality: downsample the train_pq 
    :param spark: spark session
    :param train_pq: the train data in the format of parquet
    :param val_pq: the valid data in the format of parquet
    :param sample_size: sample_users/overlap_users_of_train_&_valid
    :return: the sampled users
    '''
    train_user_uniq = train_pq.select('user_id').distinct().collect()
    val_user_uniq =  val_pq.select('user_id').distinct().collect()
    
    user_overlap = set(train_user_uniq).intersection(set(val_user_uniq))
    user_overlap = [item['user_id'] for item in user_overlap]
    
    sample_size = int(len(user_overlap) * sample_size)
    if sample_size < 1:
        raise ValueError("Sample size is smaller than 1")
    sampled_users_index = np.random.choice(len(user_overlap), sample_size)
    sampled_users=[user_overlap[index] for index in sampled_users_index]
    
    return sampled_users


def ALSmodelTrain(spark, 
                  train_pq, val_pq, 
                  model_name='als_model', 
                  use_samples=True, sample_ratio=0.01, 
                  drop_low_count_row=True, low_count_threhold=2):
    '''
    Functionality: train and select als model with grid search
    :param spark: spark session
    :param train_pq: training data
    :param val_pq: validation data
    :param model_name: user-defined model name
    :param use_samples: true if we downsample our data 
    :param sample_ratio: control downsample size
    :param drop_low_count_row: drop rows that low user feedbacks
    :param low_count_threhold: threhold for droping rows with low user feedback
    :return: a list of trained models with its model name
    '''
    ##construct the parameter sets for training the als model
    rank_ = [5, 10, 20]
    regParam_ = [0.1, 1, 10]
    alpha_ = [1, 5, 10]
    param_grid = itertools.product(rank_, regParam_, alpha_)

    ##downsample the train_data
    if use_samples:
        print("Start sampling users...")
        sampled_users = downsample(spark, train_pq, val_pq, sample_ratio)
        train_pq = train_pq[train_pq.user_id.isin(sampled_users)]
        print("Finish sampling users.")

    ##transform format of the index of train_pq from 'stirng' to 'numeric'
    print('Start transforming indices...')
    index_li = ['user_id', 'track_id']
    for index_column in index_li:
        train_pq = transformIndex(spark, index_column, train_pq)
    print('Finish transforming indices...')

    ##fit als models
    print('Start fitting als models...')
    trained_model_li=[]
    for param_tuple in param_grid:
        als = ALSmodel(spark, *param_tuple)
        model = als.fit(train_pq)
        param_str = '_'.join([str(param) for param in [*param_tuple]])
        model_name_wt_param = model_name+'_' + param_str
        model.save(model_name_wt_param)
        trained_model_li.append((model_name_wt_param, model))
        
    return trained_model_li


def ALSmodelEvaluate(spark, trained_model, val_pq, model_name, NumRecItems=20):
    '''
    Functionality: evaluate a trained als model
    :param spark: spark session
    :param trained_model: as name
    :param val_pq: validation data
    :param NumRecItems: number of recommended items for each user when we perform evaluation
    :param model_name: as name
    :return: n.a.
    '''
    ##transform format of the index of valid_pq from 'stirng' to 'numeric'
    index_li = ['user_id', 'track_id']
    for index_column in index_li:
        val_pq = transformIndex(spark, index_column, val_pq)

    ##get the true label
    true_label = val_pq.select('user_id_indexed', 'track_id_indexed') \
        .groupBy('user_id_indexed') \
        .agg(expr('collect_list(track_id_indexed) as true_item'))

    ##make predictions using the trained model
    user_id = val_pq.select('user_id_indexed').distinct()
    prediction = trained_model.recommendForUserSubset(user_id,500)
    prediction_label = prediction.select('user_id_indexed', 'recommendations.track_id_indexed')
    pred_true_rdd = prediction_label.join(F.broadcast(true_label), 
                                          'user_id_indexed', 'inner') \
                                    .rdd.map(lambda row : (row[1], row[2]))

    ##evaluate the predictions using MAP
    print('Start Evaluating...')
    map_metrics = RankingMetrics(pred_true_rdd).meanAveragePrecision
    ndcg = map_metrics.ndcgAt(NumRecItems)
    precision = map_metrics.precisionAt(NumRecItems)
    print(f'Mean Avg Precision is {map_metrics}, NDCG is {ndcg}, and Precision at top {NumRecItems} is {precision}.')


def ALSmodel(spark, rank, reg, alpha):
    '''
    Functionality: define als model
    :param spark: spark session
    :param rank: number of dimensions of latent vector
    :param reg: regularization parameter
    :param alpha: constant for comuting confidence
    :return:
    '''
    als = ALS(rank=rank, maxIter=10, regParam=reg, alpha=alpha,
              userCol="user_id_indexed", itemCol="track_id_indexed", ratingCol="count", 
              implicitPrefs=True,
              nonnegative=True, 
              coldStartStrategy="drop")
    return als

if __name__ == "__main__":
    spark = SparkSession.builder.appName('basic_als_model_train') \
                                .config('spark.executor.memory', '5g') \
                                .config('spark.driver.memory', '5g') \
                                .getOrCreate()  

    ##read the data
    RAW_DIR = 'hdfs://horton.hpc.nyu.edu/user/bm106/pub/MSD'
    train_pq_dir = os.path.join(RAW_DIR, 'cf_train.parquet')
    test_pq_dir = os.path.join(RAW_DIR, 'cf_test.parquet')
    val_pq_dir = os.path.join(RAW_DIR, 'cf_validation.parquet')
    train_pq = spark.read.parquet(train_pq_dir)
    test_pq = spark.read.parquet(test_pq_dir)
    val_pq = spark.read.parquet(val_pq_dir)

    ##train the models with the specified param sets 
    trained_model_li = ALSmodelTrain(spark, train_pq, val_pq)
    
    ##evaluate the performance of the trained models
    for trained_model_tuple in trained_model_li:
        model_name, trained_model = trained_model_tuple
        ALSmodelEvaluate(spark, trained_model, val_pq, model_name)
