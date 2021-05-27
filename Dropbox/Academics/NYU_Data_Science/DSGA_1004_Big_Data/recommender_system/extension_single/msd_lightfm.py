import argparse
import os
import pandas as pd 
import numpy as np 
from scipy.sparse import coo_matrix 
import pickle 
from tqdm import tqdm
import lightfm
from lightfm.evaluation import precision_at_k
import time 
import copy
def params():
	parser = argparse.ArgumentParser(description='lightfm')
	parser.add_argument('--data_dir', type=str, default='/scratch/work/courses/DSGA1004-2021/MSD')
	parser.add_argument('--rank', type=int, default=30)
	parser.add_argument('--verbose', type=bool, default=False)
	parser.add_argument('--valid_duration', type=int, default=10)
	parser.add_argument('--num_threads', type=int, default=1)
	parser.add_argument('--earlystop_patient', type=int, default=5)
	parser.add_argument('--epochs', type=int, default=5)
	parser.add_argument('--alpha', type=float, default=0)
	parser.add_argument('--outfile', type=str, default='res.pkl')
	parser.add_argument('--debug', type=bool, default=0)

	flags = parser.parse_args()
	return flags 

def load_parquet_and_sparse(file_dir, debug):

	#assert sample_rate == 1, "sample_rate should be 1, NotImplementError, sorry :("

	test_df = pd.read_parquet(os.path.join(file_dir, 'cf_test.parquet'))
	train_df = pd.read_parquet(os.path.join(file_dir, 'cf_train.parquet'))
	valid_df = pd.read_parquet(os.path.join(file_dir, 'cf_validation.parquet'))
	
	sizes = [len(train_df), len(valid_df), len(test_df)]
	data_df = pd.concat([train_df, valid_df, test_df])

	if debug:
		print('debug mode')
		debug_size = 10000
		data_df = data_df.sample(debug_size)
		sizes = [int(debug_size * 0.9), int(debug_size *0.05), int(debug_size * 0.05)]

	data_df['user_id'] = data_df['user_id'].astype('category').cat.codes
	data_df['track_id'] = data_df['track_id'].astype('category').cat.codes

	num_users, num_items = data_df['user_id'].max()+1,data_df['track_id'].max()+1

	train_data_coo = coo_matrix((np.ones(sizes[0]), \
		(data_df['user_id'].values[:sizes[0]], data_df['track_id'].values[:sizes[0]])),\
		shape=(num_users, num_items))
	
	valid_data_coo = coo_matrix( ( np.ones(sizes[1]), \
		(data_df['user_id'].values[sizes[0]:sizes[0]+sizes[1]], \
			data_df['track_id'].values[sizes[0]:sizes[0]+sizes[1]])), \
		shape=(num_users, num_items))

	test_data_coo = coo_matrix(( np.ones(sizes[2]), \
		(data_df['user_id'].values[sizes[0]+sizes[1]:], \
			data_df['track_id'].values[sizes[0]+sizes[1]:])), \
		shape=(num_users, num_items))

	train_weight_coo = coo_matrix(( data_df['count'].values[:sizes[0]], \
		(data_df['user_id'].values[:sizes[0]], data_df['track_id'].values[:sizes[0]])),\
		shape=(num_users, num_items))
	
	valid_weight_coo = coo_matrix(( data_df['count'].values[sizes[0]:sizes[0]+sizes[1]], \
		(data_df['user_id'].values[sizes[0]:sizes[0]+sizes[1]], \
			data_df['track_id'].values[sizes[0]:sizes[0]+sizes[1]])), \
		shape=(num_users, num_items))

	test_weight_coo = coo_matrix((data_df['count'].values[sizes[0]+sizes[1]:], \
		(data_df['user_id'].values[sizes[0]+sizes[1]:], \
			data_df['track_id'].values[sizes[0]+sizes[1]:])), \
		shape=(num_users, num_items))
	
	return {'train_coo': train_data_coo, 'valid_coo': valid_data_coo, \
	'test_coo': test_data_coo, 'train_weight': train_weight_coo, \
	'valid_weight': valid_weight_coo, \
	'test_weight': test_weight_coo,'sizes': sizes} 

class EarlyStop(object):
	"""docstring for EarlyStop"""
	def __init__(self, patient=5, ascend=False):
		super(EarlyStop, self).__init__()
		assert ~ascend, "ascent should be False, NotImplementError, sorry :<"
		self.patient = patient
		self.losses = []
		self.models = []
	def step(self, loss, model):
		self.losses.append(loss)
		self.models.append(model)
		self.models = self.models[-(self.patient+1):]

		if len(self.losses) > self.patient+1 and np.min(self.losses[-self.patient:]) >= self.losses[-(self.patient+1)]:
			return True, self.models[0],self.losses[-(self.patient+1)]

		return False, self.models[-1], self.losses[-1]


		
def train(flags, datas):

	earlystop = EarlyStop(patient = flags.earlystop_patient)

	model = lightfm.LightFM(no_components=flags.rank, k=5, n=10, learning_schedule='adagrad', \
		loss='logistic', learning_rate=0.05, rho=0.95, epsilon=1e-06, \
		item_alpha=0.0, user_alpha=0.0, max_sampled=10, random_state=None)

	for step in tqdm(range(flags.epochs // flags.valid_duration)):

		model.fit_partial(datas['train_coo'], 
			sample_weight = datas['train_coo'] + flags.alpha * datas['train_weight'] if flags.alpha != 0 else None,
			epochs=flags.valid_duration, num_threads = flags.num_threads, verbose = flags.verbose)
		model_params = copy.deepcopy(model.get_params())
		pred_k = validation(flags,model, datas['valid_coo'], datas['train_coo'])
		test_pred_k = validation(flags,model, datas['test_coo'], datas['train_coo'])
		print( "epochs %d, valid_pred_k %.8f, test_pred_k %.8f" % ((step+1) * flags.valid_duration, pred_k, test_pred_k))
		stop_signal, best_model_params, valid_loss = earlystop.step(pred_k, model_params)
		# if stop_signal:
		# 	break

	best_model_params = model_params if best_model_params is None else best_model_params
	
	#test
	model.set_params(best_model_params)
	test_score = validation(flags,model, datas['test_coo'], datas['train_coo'])

	return best_model_params, valid_loss, test_score

def validation(flags,model,valid, train):
	
	prec_k = precision_at_k(model, valid, \
			train_interactions=train, k=500, user_features=None, \
			item_features=None, preserve_rows=False, num_threads=1, check_intersections=True)
	#print(prec_k)
	prec_k = prec_k.mean()
	return prec_k


def main(flags):
	t0 = time.time()
	datas = load_parquet_and_sparse(flags.data_dir, flags.debug)
	load_time = (time.time()-t0)
	#print('loading time: %.3f' % load_time)

	t0 = time.time()
	best_model_params, valid_loss, test_score = train(flags,datas)
	train_time = time.time() - t0
	#print('train time: %.3f' % train_time)


	results = {'best_model': best_model_params, 'test_score': test_score, \
	'load_time':load_time, 'train_time': train_time}


	print('load time: %.8f, train time: %.8f, test_score: %.8f' % (load_time, train_time,test_score ))
	outfile = open(flags.outfile, 'wb')
	pickle.dump(results,outfile)
	outfile.close()

if __name__  == "__main__":
	flags = params()
	main(flags)
