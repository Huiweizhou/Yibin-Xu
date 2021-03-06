# coding: utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import os
import time
import datetime
import json
import pickle
import sys
import sklearn.metrics
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
from collections import defaultdict
import torch.nn.functional as F
import torch_geometric as tg
import networkx as nx
import gc
from collections import defaultdict,Counter
from pytorch_pretrained_bert import BertAdam,BertModel

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

#from sklearn.externals import joblib  

np.random.seed(1337)
torch.manual_seed(1337)
torch.cuda.manual_seed(1337)
#mySeed = np.random.RandomState(1234)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IGNORE_INDEX = -100
is_transformer = False

class Accuracy(object):
	def __init__(self):
		self.correct = 0
		self.total = 0
	def add(self, is_correct):
		self.total += 1
		if is_correct:
			self.correct += 1
	def get(self):
		if self.total == 0:
			return 0.0
		else:
			return float(self.correct) / self.total
	def clear(self):
		self.correct = 0
		self.total = 0 

class Config(object):
	def __init__(self, args):
		self.count_all =0
		self.count_in_train = 0
		self.acc_NA = Accuracy()
		self.acc_not_NA = Accuracy()
		self.acc_total = Accuracy()
		self.data_path = './prepro_data_bert'
		self.use_bag = False
		self.use_gpu = True
		self.is_training = True
		self.max_length = 512
		self.pos_num = 2 * self.max_length
		self.entity_num = self.max_length
		self.relation_num = 97
		self.graph_hop = 2
		self.learn_rate = 5e-6
		print('learn_rate',self.learn_rate)
		self.step_size = 2        #descrease lr for every * epoch
		self.alpha = 1.0
        
		self.train_file_size = 1    #4 for distant supervised data 
		self.test_file_size = 1
		self.max_num = 5

		self.coref_size = 20
		self.entity_type_size = 20
		self.max_epoch = 40
		self.opt_method = 'Adam'
		self.optimizer = None

		self.checkpoint_dir = './checkpoint'
		self.fig_result_dir = './fig_result'
		self.result_dir = './result'
		self.test_epoch = 2
		self.pretrain_model = self.checkpoint_dir + '/GraphCNN_multihead_bert_gate_cls_distance_hop2_checkpoint'			#path for you pretrain_model
		self.rank_path = None


		self.word_size = 128
		self.epoch_range = None
		self.cnn_drop_prob = 0.5  # for cnn
		self.keep_prob = 0.8  # for lstm

		self.period = 50

		self.batch_size = 1
		self.h_t_limit = 1800

		self.test_batch_size = self.batch_size
		self.test_relation_limit = 1800
		self.char_limit = 16
		self.sent_limit = 25
		self.dis2idx = np.zeros((1024), dtype='int64')
		self.dis2idx[1] = 1        
		self.dis2idx[2:] = 2
		self.dis2idx[4:] = 3
		self.dis2idx[8:] = 4
		self.dis2idx[16:] = 5
		self.dis2idx[32:] = 6
		self.dis2idx[64:] = 7
		self.dis2idx[128:] = 8
		self.dis2idx[256:] = 9
		self.dis2idx[512:] = 10
		self.dis_size = 20
		self.dis_num = 21
		self.dis_plus = 10

		self.train_prefix = args.train_prefix
		self.test_prefix = args.test_prefix


		if not os.path.exists("log"):
			os.mkdir("log")

	def set_data_path(self, data_path):
		self.data_path = data_path
	def set_max_length(self, max_length):
		self.max_length = max_length
		self.pos_num = 2 * self.max_length
	def set_num_classes(self, num_classes):
		self.num_classes = num_classes
	def set_window_size(self, window_size):
		self.window_size = window_size
	def set_word_size(self, word_size):
		self.word_size = word_size
	def set_max_epoch(self, max_epoch):
		self.max_epoch = max_epoch
	def set_batch_size(self, batch_size):
		self.batch_size = batch_size
	def set_opt_method(self, opt_method):
		self.opt_method = opt_method
	def set_drop_prob(self, drop_prob):
		self.drop_prob = drop_prob
	def set_checkpoint_dir(self, checkpoint_dir):
		self.checkpoint_dir = checkpoint_dir
	def set_test_epoch(self, test_epoch):
		self.test_epoch = test_epoch
	def set_pretrain_model(self, pretrain_model):
		self.pretrain_model = pretrain_model
	def set_is_training(self, is_training):
		self.is_training = is_training
	def set_use_bag(self, use_bag):
		self.use_bag = use_bag
	def set_use_gpu(self, use_gpu):
		self.use_gpu = use_gpu
	def set_epoch_range(self, epoch_range):
		self.epoch_range = epoch_range

	def from_list_to_tensor(self,item,train = True):
		new_item = {}
		new_item['document'] = torch.LongTensor(item['document'][:self.max_length])#.cuda()
		new_item['document_pos'] = torch.LongTensor(item['document_pos'][:self.max_length])#.cuda()  
		new_item['document_ner'] = torch.LongTensor(item['document_ner'][:self.max_length])#.cuda()
		graph = item['graph']

		node_num = len(graph.nodes())
		node_pos = np.zeros((node_num,len(item['document'])))   
		node_type = np.zeros(node_num)
		for node in graph.nodes():
			for position in graph.nodes[node]['exist_pos']:
				node_pos[node,position[0]:position[1]] = 1.0/(position[1]-position[0])   
			node_pos[node,:] *= 1.0/len(graph.nodes[node]['exist_pos'])   
			node_type[node] = graph.nodes[node]['type'][0]

		adj_matrix = np.zeros((node_num,node_num))
		#adj_matrix = np.eye(node_num)
		sen_matrix = np.zeros((node_num,node_num,graph.graph['max_sentence_num'],len(item['document'])))     # n*n*n_sentence*document_length
		pos_matrix_h = np.zeros((node_num,node_num,graph.graph['max_sentence_num'],len(item['document'])))   # n*n*n_sentence*document_length
		pos_matrix_t = np.zeros((node_num,node_num,graph.graph['max_sentence_num'],len(item['document'])))   # n*n*n_sentence*document_length
		for u,v,edge in graph.edges(data=True):
			adj_matrix[u,v] = 1
			for j, (sentence,position) in enumerate(zip(edge['sentences'],edge['position'])):
				sen_matrix[u,v,j,sentence[0]:sentence[1]] = 1
				for k in range(sentence[0],sentence[1]):
					delta_dis_left = k-position[0]
					delta_dis_right = k-position[1]
					if delta_dis_left < 0:
						# print (pos_matrix_h.shape)
						# print (u,v,j,k)
						pos_matrix_h[u,v,j,k] = -int(self.dis2idx[-delta_dis_left])
					elif delta_dis_right > 0:
						pos_matrix_h[u,v,j,k] = int(self.dis2idx[delta_dis_right])

					delta_dis_left = k-position[2]
					delta_dis_right = k-position[3]
					if delta_dis_left < 0:
						pos_matrix_t[u,v,j,k] = -int(self.dis2idx[-delta_dis_left])
					elif delta_dis_right > 0:
						pos_matrix_t[u,v,j,k] = int(self.dis2idx[delta_dis_right])
					
					pos_matrix_h[u,v,j,k] = pos_matrix_h[u,v,j,k] + self.dis_plus
					pos_matrix_t[u,v,j,k] = pos_matrix_t[u,v,j,k] + self.dis_plus

		# relative position
		node_relative_pos = np.zeros((node_num,node_num))
		for node_h in graph.nodes():
			for node_t in graph.nodes():
				if node_h == node_t:
					continue
				relative_pos = graph.nodes[node_h]['exist_pos'][0][0] - graph.nodes[node_t]['exist_pos'][0][0]
				if relative_pos < 0:
					node_relative_pos[node_h,node_t] = -self.dis2idx[-relative_pos]
				else:
					node_relative_pos[node_h,node_t] = self.dis2idx[relative_pos]

		new_item['adj_matrix'] = torch.FloatTensor(adj_matrix)#.cuda()
		new_item['sen_matrix'] = torch.BoolTensor(sen_matrix[:,:,:self.max_num,:self.max_length])#.cuda()
		new_item['pos_matrix_h'] = torch.LongTensor(pos_matrix_h[:,:,:self.max_num,:self.max_length])#.cuda()
		new_item['pos_matrix_t'] = torch.LongTensor(pos_matrix_t[:,:,:self.max_num,:self.max_length])#.cuda()
		new_item['node_pos'] = torch.FloatTensor(node_pos[:,:self.max_length])#.cuda()
		new_item['node_type'] = torch.LongTensor(node_type)#.cuda()
		new_item['node_relative_pos'] = torch.LongTensor(node_relative_pos)
		new_item['title'] = item['title']

		new_item['labels'] = item['labels']

		label_matrix = item['label_matrix']			#numpy
		new_item['label_matrix'] = torch.FloatTensor(label_matrix)#.cuda()
		
		self.id2rel = {v: k for k,v in self.rel2id.items()}
		if not train:
			vertexSet = item['vertexSet']
			label_set = {}
			for i in range(new_item['label_matrix'].size(0)):
				entity_i = vertexSet[i][0]['name']
				for j in range(new_item['label_matrix'].size(1)):
					entity_j = vertexSet[j][0]['name']
					for r in range(new_item['label_matrix'].size(2)):
						relation_id =  self.id2rel[r]
						self.count_all += 1
						if (entity_i,entity_j,relation_id) in self.fact_in_train:
							self.count_in_train += 1
							label_set[(i,j,r)] = True
						else:
							label_set[(i,j,r)] = False
			new_item['label_set'] = label_set


		new_item['label_mask'] = item['label_mask']		

		
		
		return new_item#,length_counter

	def load_train_data(self):
		print("Reading training data...")
		prefix = self.train_prefix
		print ('train', prefix)		#dev_train

		#self.data_word_vec = np.load(os.path.join(self.data_path, 'vec.npy'))
		self.data_char_vec = np.load(os.path.join(self.data_path, 'char_vec.npy'))
		self.rel2id = json.load(open(os.path.join(self.data_path, 'rel2id.json')))
		self.id2rel = {v: k for k,v in self.rel2id.items()}

		self.train_file = []
		for i in range (self.train_file_size):
		    self.train_file.extend(pickle.load(open(os.path.join(self.data_path, prefix + '_' +str(i) + '.pkl'),'rb')))   
		if self.rank_path != None:
			self.rank = pickle.load(open(self.rank_path,'rb'))
		print("Finish reading")

		self.train_len = ins_num = len(self.train_file)
		self.train_order = list(range(ins_num))
		self.train_batches = ins_num // self.batch_size
		if ins_num % self.batch_size != 0:
			self.train_batches += 1

	def load_test_data(self):
		print("Reading testing data...")

		prefix = self.test_prefix
		print (prefix)
		self.is_test = ('dev_test' == prefix)
		if self.is_test:
			#self.data_word_vec = np.load(os.path.join(self.data_path, 'vec.npy'))
			self.data_char_vec = np.load(os.path.join(self.data_path, 'char_vec.npy'))
			self.rel2id = json.load(open(os.path.join(self.data_path, 'rel2id.json')))
			self.id2rel = {v: k for k,v in self.rel2id.items()}

		self.test_file = []
		for i in range (self.test_file_size):
			self.test_file = pickle.load(open('./prepro_data_bert/new/dev_dev_low2high_0.pkl','rb'))		#dev data
			# self.test_file = pickle.load(open('./prepro_data_bert/new/dev_test_low2high_0.pkl','rb'))		#test data

		self.test_len = len(self.test_file)

		print("Finish reading")

		self.test_batches = self.test_len // self.test_batch_size
		if self.test_len % self.test_batch_size != 0:
			self.test_batches += 1
		self.test_order = list(range(self.test_len))

	def gen_train_facts_anno(self):		
		data_file_name = '../data/train_annotated.json'

		fact_in_train = set([])   
		ori_data = json.load(open(data_file_name))
		for data in ori_data:
			vertexSet = data['vertexSet']
			for label in data['labels']:
				rel = label['r']
				for n1 in vertexSet[label['h']]:
					for n2 in vertexSet[label['t']]:
						fact_in_train.add((n1['name'], n2['name'], rel))

		self.fact_in_train = fact_in_train
		print('fact in train_annotated have been loading...')

	def gen_train_facts_distant(self):		
		data_file_name = '../data/train_distant.json'

		fact_in_train = set([])  
		ori_data = json.load(open(data_file_name))
		for data in ori_data:
			vertexSet = data['vertexSet']
			for label in data['labels']:
				rel = label['r']
				for n1 in vertexSet[label['h']]:
					for n2 in vertexSet[label['t']]:
						fact_in_train.add((n1['name'], n2['name'], rel))
		self.fact_in_train = fact_in_train
		print('fact in train_distant have been loading...')
	
	def train(self, model_pattern, model_name):
		print('new_train...6_20')
		ori_model = model_pattern(config = self)
		if self.pretrain_model != None:                  
			ori_model.load_state_dict(torch.load(self.pretrain_model))
			print('load pretrain_model',self.pretrain_model)
		ori_model.cuda()
		model = ori_model

		optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr = self.learn_rate)#,weight_decay=1e-6)  
		BCE = nn.BCELoss(reduction='mean')#,weight = torch.FloatTensor(label_weights).cuda())          

		if not os.path.exists(self.checkpoint_dir):
			os.mkdir(self.checkpoint_dir)

		best_auc = 0.0
		best_f1 = 0.0
		best_epoch = 0
		best_ign_f1 = 0.0
		best_ign_auc = 0.0
		start_time = time.time()
		model.train()

		global_step = 0
		total_loss = 0

		def logging(s, print_=True, log_=True):                 #
			if print_:
				print(s)
			if log_:
				with open(os.path.join(os.path.join("log", model_name)), 'a+') as f_log:
					f_log.write(s + '\n')

		plt.xlabel('Recall')
		plt.ylabel('Precision')
		plt.ylim(0.2, 1.0)
		plt.xlim(0.0, 0.8)
		plt.title('Precision-Recall')
		plt.grid(True)

		for epoch in range(self.max_epoch):

			self.acc_NA.clear()
			self.acc_not_NA.clear()
			self.acc_total.clear()
			total_loss = Variable(torch.Tensor([0]).cuda(), requires_grad=True)
			all_loss = 0.0
			logging('-' * 79)
			for i_data in tqdm(range(len(self.train_file))): 
				if i_data == 1985:
					continue
				index_data = i_data
				data = self.from_list_to_tensor(self.train_file[index_data],train= True)

				#path_counter.update(counter)
				#continue
				document = data['document'].cuda()
				#document_char = data['document_char'].cuda()
				document_ner = data['document_ner'].cuda()
				document_pos = data['document_pos'].cuda()
				adj_matrix = data['adj_matrix'].cuda()
				sen_matrix = data['sen_matrix'].cuda()
				pos_matrix_h = data['pos_matrix_h'].cuda()
				pos_matrix_t = data['pos_matrix_t'].cuda()
				node_pos = data['node_pos'].cuda()   
				node_type = data['node_type'].cuda()
				label_matrix = data['label_matrix'].cuda()
				node_relative_pos = data['node_relative_pos'].cuda()
				predict_re = model(document,document_ner,document_pos,adj_matrix,sen_matrix,pos_matrix_h,pos_matrix_t,node_pos,node_type,node_relative_pos)
				predict_re = torch.sigmoid(predict_re)
				temp_loss = Variable(torch.Tensor([0]).cuda(), requires_grad=True)
				node_num = label_matrix.size(0)
				for h_i in range(node_num):
					for t_j in range(node_num):
						if h_i == t_j:
							continue
						loss = BCE(predict_re[h_i][t_j], label_matrix[h_i][t_j])
						temp_loss = temp_loss + loss
				temp_loss = temp_loss/(node_num*node_num-node_num)

				total_loss = torch.add(total_loss,temp_loss)

				if global_step % self.batch_size == 0:
					all_loss += total_loss.data.cpu().numpy()
					total_loss = total_loss / self.batch_size
					optimizer.zero_grad()
					total_loss.backward()
					optimizer.step()
					total_loss = Variable(torch.Tensor([0]).cuda(), requires_grad=True)

				output = torch.sigmoid(predict_re).data.cpu().numpy()
				labels = label_matrix.data.cpu().numpy()

				for i in range(output.shape[0]):
					for j in range(output.shape[1]):
						if i==j:
							continue
						true_label = labels[i,j]
						r = output[i,j].argmax()
						if true_label[0] == 1:
							self.acc_NA.add(true_label[r] == 1)
						else:
							self.acc_not_NA.add(true_label[r] == 1)

						self.acc_total.add(true_label[r] == 1)

				global_step += 1
			
			#logging('-' * 69)
			elapsed = time.time() - start_time
			print (self.acc_NA.correct,self.acc_NA.total)
			print (self.acc_not_NA.correct,self.acc_not_NA.total)
			logging('| epoch {:2d} | ms/b {:5.2f} | train loss {:5.3f} | NA acc: {:4.2f} | not NA acc: {:4.2f}  | tot acc: {:4.2f} '.format(epoch, elapsed * 1000 / len(self.train_file), float(all_loss), self.acc_NA.get(), self.acc_not_NA.get(), self.acc_total.get()))
			total_loss = 0
			model.eval()
			f1, auc, pr_x, pr_y, ign_f1, ign_auc = self.test(model, model_name)
			model.train()
			start_time = time.time()
			#logging('-' * 69)


			if f1 > best_f1:
				best_f1 = f1
				best_auc = auc
				best_epoch = epoch
				best_ign_f1 = ign_f1
				best_ign_auc = ign_auc
				path = os.path.join(self.checkpoint_dir, model_name)
				torch.save(ori_model.state_dict(), path)

				plt.plot(pr_x, pr_y, lw=2, label=str(epoch))
				plt.legend(loc="upper right")
				plt.savefig(os.path.join("fig_result", model_name))
			
			del pr_x,pr_y
			print ('garbage',gc.collect())

		print("Finish training")
		print("Best epoch = %d | auc = %f" % (best_epoch, best_auc))
		print("Storing best result...")
		print("Finish storing")

	def test(self, model, model_name, output=False, input_theta=-1):
		data_idx = 0
		eval_start_time = time.time()
		test_result_ignore = []
		total_recall_ignore = 0

		test_result = []
		total_recall = 0
		total_correct = 0
		na_recall = 0
		na_correct = 0
		top1_acc = have_label = 0
		index = 0


		def logging(s, print_=True, log_=True):
			if print_:
				print(s)
			if log_:
				with open(os.path.join(os.path.join("log", model_name)), 'a+') as f_log:
					f_log.write(s + '\n')

		for data in tqdm(self.test_file): 
			
			data = self.from_list_to_tensor(data,False)
			document = data['document'].cuda()
			document_ner = data['document_ner'].cuda()
			document_pos = data['document_pos'].cuda()
			#title = data['title'].cuda()
			#title_char = data['title_char'].cuda()
			adj_matrix = data['adj_matrix'].cuda()
			sen_matrix = data['sen_matrix'].cuda()
			pos_matrix_h = data['pos_matrix_h'].cuda()
			pos_matrix_t = data['pos_matrix_t'].cuda()
			node_pos = data['node_pos'].cuda()
			node_type = data['node_type'].cuda()
			#label_mask = data['label_mask']
			#title = data['title']
			label_matrix = data['label_matrix'].cuda()
			node_relative_pos = data['node_relative_pos'].cuda()

			label_set = data['label_set']
			with torch.no_grad():
				predict_re = model(document,document_ner,document_pos,adj_matrix,sen_matrix,pos_matrix_h,pos_matrix_t,node_pos,node_type,node_relative_pos)

			predict_re = torch.sigmoid(predict_re)

			predicts = predict_re.data.cpu().numpy()
			labels = label_matrix.data.cpu().numpy()

			for i in range(predicts.shape[0]):	
				for j in range(predicts.shape[1]):
					if i == j:
						continue
					r = np.argmax(predicts[i, j])

					if labels[i,j,r]:
						top1_acc += 1
					flag = False

					in_train = False
					for k in range (self.relation_num):
						if k == 0:
							if labels[i,j,k]:
								na_recall += 1
								if r==0:
									na_correct += 1
							continue
						if labels[i,j,k]:
							total_recall += 1
							if r == k:
								total_correct += 1
							flag = True
							if label_set[(i,j,k)] == True:
								in_train = True
						if in_train:
							test_result_ignore.append (( labels[i,j,k], float(predicts[i,j,k]),index,i,j,k))
						test_result.append(( labels[i,j,k], float(predicts[i,j,k]),in_train,index,i,j,k))#, self.id2rel[k], i, j, k) )

					if flag:
						have_label += 1
			index += 1
		
		print (na_correct,na_recall)
		print('total_correct',total_correct)
		print ('total_recall',total_recall)
		print('| test step | time: {:5.2f}'.format((time.time() - eval_start_time)))

		print('count_all',self.count_all)
		print('count_in_train',self.count_in_train)
		self.count_all =0
		self.count_in_train = 0

		test_result.sort(key = lambda x: x[1], reverse=True)
		max_n = 1000000
		test_result = test_result[:min(max_n,len(test_result))]

		pr_x = []
		pr_y = []
		correct = 0
		w = 0	

		if total_recall == 0:
			total_recall = 1 

		for i, item in enumerate(test_result):
			correct += item[0]
			pr_y.append(float(correct) / (i + 1))
			pr_x.append(float(correct) / total_recall)
			if item[1] > input_theta:
				w = i

		pr_x = np.asarray(pr_x, dtype='float32')
		pr_y = np.asarray(pr_y, dtype='float32')
		f1_arr = (2 * pr_x * pr_y / (pr_x + pr_y + 1e-20))
		f1 = f1_arr.max()
		f1_pos = f1_arr.argmax()
		p = pr_x[f1_pos]
		r = pr_y[f1_pos]
		theta = test_result[f1_pos][1]

		if input_theta==-1:
			w = f1_pos
			input_theta = theta

		auc = sklearn.metrics.auc(x = pr_x, y = pr_y)
		if not self.is_test:
			logging('ALL  : Theta {:3.4f} | P {:3.4f} | R {:3.4f} | F1 {:3.4f} | AUC {:3.4f}'.format(theta, p, r, f1, auc))
		else:
			logging('ma_f1 {:3.4f} | input_theta {:3.4f} test_result F1 {:3.4f} | AUC {:3.4f}'.format(f1, input_theta, f1_arr[w], auc))
		output = True
		if output:
			output = [{'index': x[3], 'h_idx': x[4], 't_idx': x[5], 'r_idx': x[6], 'r': self.id2rel[x[6]], 'title': self.test_file[x[3]]['title']} for x in test_result[:w+1]]
			json.dump(output, open(os.path.join(self.result_dir, model_name + self.test_prefix + "_index.json"), "w"))
		if not os.path.exists(self.fig_result_dir):
			os.mkdir(self.fig_result_dir)

		pr_x = []
		pr_y = []
		correct = correct_in_train = 0
		w = 0
		for i, item in enumerate(test_result):
			correct += item[0]
			if (int(item[0]) & item[2]):
				correct_in_train += 1
			if correct_in_train==correct:
				p = 0
			else:
				p = float(correct - correct_in_train) / (i + 1 - correct_in_train)
			pr_y.append(p)
			pr_x.append(float(correct) / total_recall)
			if item[1] > input_theta:
				w = i

		pr_x = np.asarray(pr_x, dtype='float32')
		pr_y = np.asarray(pr_y, dtype='float32')
		f1_arr = (2 * pr_x * pr_y / (pr_x + pr_y + 1e-20))
		ign_f1 = f1_arr.max()

		ign_auc = sklearn.metrics.auc(x = pr_x, y = pr_y)
		print('ign_f1',ign_f1)
		print('ign_auc',ign_auc)

		del test_result

		return f1, auc, pr_x, pr_y , ign_f1 , ign_auc



	def testall(self, model_pattern, model_name, input_theta):#, ignore_input_theta):
		print('input_theta:',input_theta)
		# 0.2782
		model = model_pattern(config = self)

		model.load_state_dict(torch.load(os.path.join(self.checkpoint_dir, model_name)))	#self.checkpoint_dir = './checkpoint'

		model.cuda()
		model.eval()
		f1, auc, pr_x, pr_y ,ign_f1 , ign_auc= self.test(model, model_name, True, input_theta)
		print('f1:',f1)
		print('auc:',auc)
		print('ign_f1:',ign_f1)
		print('ign_auc:',ign_auc)
