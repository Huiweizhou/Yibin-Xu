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
# import torch_geometric as tg
import networkx as nx
import gc
from collections import defaultdict,Counter
from pytorch_pretrained_bert import BertAdam,BertModel

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

#from sklearn.externals import joblib   #解决pickle不能存储大数据的问题

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
		self.acc_NA = Accuracy()
		self.acc_not_NA = Accuracy()
		self.acc_total = Accuracy()
		self.data_path = './prepro_data_bert/gda_and_cdr'
		self.use_bag = False
		self.use_gpu = True
		self.is_training = True
		self.max_length = 512
		self.pos_num = 2 * self.max_length
		self.entity_num = self.max_length
		self.relation_num = 2
		self.graph_hop = 0
		# self.learn_rate = 1e-5
		self.learn_rate = 1e-5
		self.step_size = 2        #descrease lr for every * epoch
		self.alpha = 1.0
		
		self.train_file_size = 6    #远程监督训练语料是4
		self.test_file_size = 1
		# self.max_sentence_length = 197    #最长的句子的长度
		# self.max_mention_num = 36    #实体对所对应的最多的mention对个数
		# self.max_sentence_num = 37
		self.max_num = 10			#好像是一条边上的最大的句子数量

		self.coref_size = 20
		self.entity_type_size = 20
		self.max_epoch = 40
		self.opt_method = 'Adam'
		self.optimizer = None

		self.checkpoint_dir = './checkpoint'
		self.fig_result_dir = './fig_result'
		self.result_dir = './result'
		self.test_epoch = 2			#没啥用
		# self.pretrain_model = self.checkpoint_dir +'/GraphCNN_multihead_bert_distance_checkpoint'
		# self.pretrain_model = self.checkpoint_dir +'/GraphCNN_multihead_bert_distance_hop2_curriculum_low2high_rank_checkpoint'
		self.pretrain_model = None
		# self.pretrain_model = self.checkpoint_dir +'/GraphCNN_multihead_bert_gate_distance_hop2_curriculum_low2high_rank_checkpoint'
		# self.pretrain_model = self.checkpoint_dir + '/GraphCNN_multihead_bert_gate_retrain_meanwhile_distance_low2high_hop2_checkpoint'
		# self.pretrain_model = self.checkpoint_dir + '/GraphCNN_multihead_bert_gate_distance_low2high_hop2_checkpoint'
		# self.pretrain_model = self.checkpoint_dir + '/GraphCNN_multihead_bert_gate_cls_distance_hop2_checkpoint'
		# self.pretrain_model = self.checkpoint_dir + '/GraphCNN_multihead_bert_gate_cls_for_CDR_6_19'
		#self.rank_path = self.result_dir + '/GraphCNN_multihead_bert_afterpretrain_hop2_rank_checkpointdev_train_rank.pkl'
		self.rank_path = None
		# self.ner2id = {'C':0,'D':1}
		self.ner2id = {'G':0,'D':1}


		self.word_size = 128
		self.epoch_range = None
		self.cnn_drop_prob = 0.5  # for cnn
		self.keep_prob = 0.8  # for lstm

		self.period = 50

		self.batch_size = 1		#原先是1
		self.h_t_limit = 1800

		self.test_batch_size = self.batch_size
		self.test_relation_limit = 1800
		self.char_limit = 16
		self.sent_limit = 25
		self.dis2idx = np.zeros((1024), dtype='int64')
		self.dis2idx[1] = 1         #衰减的相对位置？？？
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
			os.mkdir("log")		#如果目录下没有log文件夹，创建log文件夹

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
		# new_data = []
		# for item in tqdm(data):
		# length_counter = Counter()
		new_item = {}
		new_item['document'] = torch.LongTensor(item['document'][:self.max_length])#.cuda()
		# new_item['document_char'] = torch.LongTensor(item['document_char'][:self.max_length])#.cuda()
		# new_item['title'] = torch.LongTensor(item['title'][:self.max_length])#.cuda()
		# new_item['title_char'] = torch.LongTensor(item['title_char'][:self.max_length])#.cuda()
		new_item['document_pos'] = torch.LongTensor(item['document_pos'][:self.max_length])#.cuda()    #我的算法好像没用了，这个
		new_item['document_ner'] = torch.LongTensor(item['document_ner'][:self.max_length])#.cuda()
		graph = item['graph']
		pubID = item['ID']
		# print('pubID',pubID)

		node_num = len(graph.nodes())
		node_pos = np.zeros((node_num,len(item['document'])))   #节点特征，把所有mention都赋值为1
		node_type = np.zeros(node_num)
		for node in graph.nodes():
			for position in graph.nodes[node]['exist_pos']:
				#print (position[0],position[1])
				node_pos[node,position[0]:position[1]] = 1.0/(position[1]-position[0])   #平均 乘以长度为系数
			node_pos[node,:] *= 1.0/len(graph.nodes[node]['exist_pos'])    #平均，除以总次数
			if graph.nodes[node]['type'] == 'C':
				graph.nodes[node]['type'] = 'G'
			node_type[node] = self.ner2id[graph.nodes[node]['type']]

		adj_matrix = np.zeros((node_num,node_num))
		#adj_matrix = np.eye(node_num)
		sen_matrix = np.zeros((node_num,node_num,graph.graph['max_sentence_num'],len(item['document'])))     # n*n*n_sentence*document_length
		# sen_matrix = np.zeros((node_num,node_num,len(item['document'])))     # n*n*document_length
		# pos_matrix_h = np.zeros((node_num,node_num,graph.graph['max_sentence_num'],graph.graph['max_sentence_length']))   # n*n*n_sentence*document_length
		# pos_matrix_t = np.zeros((node_num,node_num,graph.graph['max_sentence_num'],graph.graph['max_sentence_length']))   # n*n*n_sentence*document_length
		pos_matrix_h = np.zeros((node_num,node_num,graph.graph['max_sentence_num'],len(item['document'])))   # n*n*n_sentence*document_length
		pos_matrix_t = np.zeros((node_num,node_num,graph.graph['max_sentence_num'],len(item['document'])))   # n*n*n_sentence*document_length
		# pos_matrix_h = np.zeros((node_num,node_num,len(item['document'])))   # n*n*document_length
		# pos_matrix_t = np.zeros((node_num,node_num,len(item['document'])))   # n*n*document_length
		label_mask = []
		relation = item['relation']
		#print (relation)
		for u,v,edge in graph.edges(data=True):
			adj_matrix[u,v] = 1
			for j, (sentence,position) in enumerate(zip(edge['sentences'],edge['position'])):
				# print (sen_matrix.shape)
				# print (u,v,j,sentence[0],sentence[1])
				sen_matrix[u,v,j,sentence[0]:sentence[1]] = 1
				# sen_matrix[u,v,k,sentence[0]:sentence[1]] = 1
				# sen_matrix[u,v,k,position[0]:position[1]] = 0
				# sen_matrix[u,v,k,position[2]:position[3]] = 0
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
		# print('relation',relation)
		# print('graph.nodes',graph.nodes())
		for node_h in graph.nodes():
			for node_t in graph.nodes():
				if node_h == node_t:
					continue
				relative_pos = graph.nodes[node_h]['exist_pos'][0][0] - graph.nodes[node_t]['exist_pos'][0][0]
				if relative_pos < 0:
					node_relative_pos[node_h,node_t] = -self.dis2idx[-relative_pos]
				else:
					node_relative_pos[node_h,node_t] = self.dis2idx[relative_pos]
				if (graph.nodes[node_h]['type'] == 'G' or graph.nodes[node_h]['type'] == 'C')  and graph.nodes[node_t]['type'] == 'D':
					if (node_h,node_t) in relation:
						label_mask.append((node_h,node_t,1))
						#print (node_h,node_t)
					else:
						label_mask.append((node_h,node_t,0))
		# print('label mask',label_mask)

		new_item['adj_matrix'] = torch.FloatTensor(adj_matrix)#.cuda()
		# new_item['sen_matrix'] = torch.BoolTensor(sen_matrix[:,:,:self.max_num,:self.max_length])#.cuda()
		new_item['sen_matrix'] = torch.ByteTensor(sen_matrix[:,:,:self.max_num,:self.max_length])#.cuda()
		new_item['pos_matrix_h'] = torch.LongTensor(pos_matrix_h[:,:,:self.max_num,:self.max_length])#.cuda()
		new_item['pos_matrix_t'] = torch.LongTensor(pos_matrix_t[:,:,:self.max_num,:self.max_length])#.cuda()
		new_item['node_pos'] = torch.FloatTensor(node_pos[:,:self.max_length])#.cuda()
		new_item['node_type'] = torch.LongTensor(node_type)#.cuda()
		new_item['node_relative_pos'] = torch.LongTensor(node_relative_pos)
		#new_item['title'] = item['title']
		#new_item['relation'] = item['relation']

		# if train:
		# 	label_matrix = np.zeros((node_num,node_num,self.relation_num))
		# 	for label in item['labels']:
		# 		rel = label['r']
		# 		assert(rel in self.rel2id)
		# 		label['r'] = self.rel2id[label['r']]        #关系对应的id
		# 		label_matrix[label['h'], label['t'],label['r']] = 1     #正例对应的位置是label id else 0 (Na)
		# 	for h_i in range (node_num):
		# 		for t_j in range (node_num):
		# 			label_sum = label_matrix[h_i,t_j,:]
		# 			if label_sum.sum() == 0:
		# 				label_matrix[h_i,t_j,0] = 1

		# 	new_item['label_matrix'] = torch.BoolTensor(label_matrix)
		#label_matrix = item['label_matrix']
		# na_num = np.sum(label_matrix[:,:,0])-node_num  #NA relation num
		# na_sum = min(na_num,(node_num*(node_num-1) - na_num)*3)
		# if na_sum:
		# 	drop_ratio = float(na_sum) / na_num         #保留的na/全部na，以这个概率保留
		# else:
		# 	drop_ratio = 0.5
		#label_mask = []
		# for h_i in range(node_num):
		# 	for t_j in range (node_num):
		# 		# if h_i == t_j:
		# 			#label_mask[h_i,t_j] = 0
		# 		# 	continue
		# 		# try:
		# 		#	length = nx.shortest_path_length(graph, source=h_i, target=t_j)
		# 		# except:
		# 		#	length = 0
		# 		# if length and length<=3:
		# 		# if label_matrix[h_i,t_j,0] == 1 and np.random.uniform(0,1) >= drop_ratio:
		# 		#	continue
		# 		# label_mask.append((h_i,t_j))
		# 		# if label_matrix[h_i,t_j,0] == 1:
		# 		# 	if np.random.uniform(0,1)>= drop_ratio:
		# 		# 		label_mask[h_i,t_j] = 0
		# 		for r_k in range (1,self.relation_num):
		# 			if label_matrix[h_i,t_j,r_k] == 1:
		# 				try:
		# 					length = nx.shortest_path_length(graph, source=h_i, target=t_j)
		# 				except:
		# 					length = 0
		# 				length_counter[length] += 1
		# if not len(label_mask):
		# 	print (label_matrix[:,:,0])
		# 	print (node_num,na_num,na_sum,drop_ratio)

		# na_num = np.sum(label_matrix[:,:,0])  #非0的正样例个数
		# na_sum = min(na_num,(node_num*(node_num-1) - na_num)*3)
		# drop_ratio = float(na_sum) / na_num         #保留的na/全部na，以这个概率保留
		# label_mask = np.ones((node_num,node_num))
		# for h_i in range(node_num):
		# 	for t_j in range (node_num):
		# 		if h_i == t_j:
		# 			label_mask[h_i,t_j] = 0
		# 			continue
		# 		if label_matrix[h_i,t_j,0] == 1:
		# 			if np.random.uniform(0,1)>= drop_ratio:
		# 				label_mask[h_i,t_j] = 0
		#new_item['label_mask'] = torch.FloatTensor(label_mask)
		
		#find the shortest path
		# label_mask = np.zeros((node_num,node_num))
		# for h_i in range(node_num):
		# 	for t_j in range (node_num):
		# 		try:
		# 			length = nx.shortest_path_length(graph, source=h_i, target=t_j)
		# 		except:
		# 			length = 0
		# 		label_mask[h_i,t_j] = min(length,self.graph_hop-1)
		# new_item['label_mask'] = label_mask
		#new_item['label_matrix'] = torch.FloatTensor(label_matrix)#.cuda()
		new_item['label_mask'] = label_mask
		#print (relation)
		#print (label_mask)

			# new_data.append(new_item)
		
		return new_item#,length_counter

	def load_train_data(self):
		print('*'*20)
		print("Reading training data...")
		prefix = self.train_prefix
		print ('train prefix:', prefix)		#train_train

		#self.data_word_vec = np.load(os.path.join(self.data_path, 'vec.npy'))
		#self.data_char_vec = np.load(os.path.join(self.data_path, 'char_vec.npy'))
		#self.rel2id = json.load(open(os.path.join(self.data_path, 'rel2id.json')))
		#self.id2rel = {v: k for k,v in self.rel2id.items()}

		# self.data_train_word = np.load(os.path.join(self.data_path, prefix+'_word.npy'))
		# self.data_train_pos = np.load(os.path.join(self.data_path, prefix+'_pos.npy'))
		# self.data_train_ner = np.load(os.path.join(self.data_path, prefix+'_ner.npy'))
		# self.data_train_char = np.load(os.path.join(self.data_path, prefix+'_char.npy'))

		self.train_file = []
		# for i in range (self.train_file_size):
		#     self.train_file.extend(pickle.load(open(os.path.join(self.data_path, prefix + '_' +str(i) + '.pkl'),'rb')))   #load pickle文件，文件过大
		for i in range (self.train_file_size):
			self.train_file.extend(pickle.load(open(os.path.join(self.data_path, prefix +str(i) + '.pkl'),'rb')))   #load pickle文件，文件过大
		if self.rank_path != None:
			self.rank = pickle.load(open(self.rank_path,'rb'))
		# self.train_file = self.from_list_to_tensor(self.train_file)
		print("Finish reading train data!")

		self.train_len = ins_num = len(self.train_file)
		assert(self.train_len==len(self.train_file))
		print('训练集样例个数：', self.train_len)

		train_rank = [i for i in range(ins_num)]		#用来打乱训练顺序
		random.shuffle(train_rank)
		self.train_rank = train_rank

		self.train_order = list(range(ins_num))
		self.train_batches = ins_num // self.batch_size
		if ins_num % self.batch_size != 0:
			self.train_batches += 1
		print('*'*20)

	def load_test_data(self):
		print("Reading testing data...")

		prefix = self.test_prefix
		print ('test prefix:',prefix)		#test_test
		# self.is_test = ('dev_test' == prefix)
		#if self.is_test:
			#self.data_word_vec = np.load(os.path.join(self.data_path, 'vec.npy'))
			#self.data_char_vec = np.load(os.path.join(self.data_path, 'char_vec.npy'))
			#self.rel2id = json.load(open(os.path.join(self.data_path, 'rel2id.json')))
			#self.id2rel = {v: k for k,v in self.rel2id.items()}

		self.test_file = []
		# for i in range (self.test_file_size):
		#     self.test_file.extend(pickle.load(open(os.path.join(self.data_path, prefix + '_' +str(i) + '.pkl'),'rb')))   #load pickle文件，文件过大
		for i in range (self.test_file_size):
			self.test_file.extend(pickle.load(open(os.path.join(self.data_path, prefix +str(i) + '.pkl'),'rb')))   #load pickle文件，文件过大
		# self.test_file = self.from_list_to_tensor(self.test_file)
		# print("Finish reading")

		self.test_len = len(self.test_file)
		print('测试集样例个数：', self.test_len)
		#assert(self.test_len==len(self.test_file))

		print("Finish reading")

		self.test_batches = self.test_len // self.test_batch_size
		if self.test_len % self.test_batch_size != 0:
			self.test_batches += 1

		self.test_order = list(range(self.test_len))
		# self.test_order.sort(key=lambda x: np.sum(self.data_test_word[x] > 0), reverse=True)
	
	def train(self, model_pattern, model_name):

		print('train_both on gda and cdr, test on gda.')
		ori_model = model_pattern(config = self)
		# if self.pretrain_model != None:                   #导入预训练模型,此处为none，不导入预训练模型
		# 	print('load pretrain_model:',self.pretrain_model)
		# 	ori_model.load_state_dict(torch.load(self.pretrain_model))
		ori_model.cuda()
		model = ori_model
		# for name,parameters in ori_model.named_parameters():
		# 	print(name,':',parameters.size())
		# bert = BertModel.from_pretrained('./bert/bert-base-uncased/')
		#bert.cuda()
		#model = nn.DataParallel(ori_model)                  #多GPU并行？ 不太懂
		# parameters = []
		# for param in model.parameters():
		# 	parameters.append(param)
		# for param in bert.parameters():
		# 	parameters.append(param)
		optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr = self.learn_rate)#,weight_decay=1e-6)   #只放required_grad
		#optimizer = BertAdam(filter(lambda p: p.requires_grad, model.parameters()),lr = self.learn_rate,weight_decay=1e-6)   #只放required_grad
		#StepLR = torch.optim.lr_scheduler.StepLR(optimizer, step_size = self.step_size, gamma=0.9)
		# nll_average = nn.CrossEntropyLoss(size_average=True, ignore_index=IGNORE_INDEX)     #单标签损失专用
		#BCE = nn.BCEWithLogitsLoss(reduction='sum')              #多标签损失专用
		#BCE_mean = nn.BCEWithLogitsLoss(reduction='mean')              #多标签损失专用
		
		# label_weights = np.ones(self.relation_num)
		# label_weights[0] = 0.2
		CE = nn.CrossEntropyLoss(reduction='mean')#,weight = torch.FloatTensor(label_weights).cuda())              #多标签损失专用

		if not os.path.exists(self.checkpoint_dir):
			os.mkdir(self.checkpoint_dir)

		best_auc = 0.0
		best_f1 = 0.0
		best_epoch = 0
		start_time = time.time()
		model.train()
		#bert.train()

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

		# P,R,F = self.test(model, model_name)		#debug
		# print('P:',P,'R:',R,'F:',F)
		for epoch in range(self.max_epoch):
			global_step = 0

			self.acc_NA.clear()
			self.acc_not_NA.clear()
			self.acc_total.clear()
			total_loss = Variable(torch.Tensor([0]).cuda(), requires_grad=True)
			all_loss = 0.0
			logging('-' * 79)
			train_rank = self.train_rank
			random.shuffle(train_rank)
			print('第%d个epoch的训练顺序已经打乱'%epoch)
			#path_counter = Counter()
			for i_data in tqdm(range(len(self.train_file))):   #一个batch的数据
				#index_data = self.rank[i_data]
				#data = self.from_list_to_tensor(self.train_file[-index_data-1])
				# index_data = i_data
				index_data = train_rank[i_data]
				data = self.from_list_to_tensor(self.train_file[index_data])

				#path_counter.update(counter)
				#continue
				document = data['document'].cuda()
				#document_char = data['document_char'].cuda()
				document_ner = data['document_ner'].cuda()
				document_pos = data['document_pos'].cuda()
				#title = data['title'].cuda()
				#title_char = data['title_char'].cuda()
				adj_matrix = data['adj_matrix'].cuda()
				sen_matrix = data['sen_matrix'].cuda()
				pos_matrix_h = data['pos_matrix_h'].cuda()
				pos_matrix_t = data['pos_matrix_t'].cuda()
				node_pos = data['node_pos'].cuda()   #节点特征 最重要
				node_type = data['node_type'].cuda()
				label_mask = data['label_mask']
				if len(label_mask) ==0:		#有一些样例有问题，relations（label）是空的
					continue
				node_relative_pos = data['node_relative_pos'].cuda()
				#title = data['title']
				#label_mask = data['label_mask']
				#label_mask = data['label_mask'].cuda()
				#document,_ = bert(document.unsqueeze(0),output_all_encoded_layers = False)

				#predict_re = model(document,document_char,document_ner,document_pos,title,title_char,adj_matrix,sen_matrix,pos_matrix_h,pos_matrix_t,node_pos)
				predict_re = model(document,document_ner,document_pos,adj_matrix,sen_matrix,pos_matrix_h,pos_matrix_t,node_pos,node_type,node_relative_pos)
				#predict_re = torch.softmax(predict_re,dim = -1)
				#predict_re = predict_re * label_mask.unsqueeze(2).expand(-1,-1,self.relation_num)
				#label_matrix = label_matrix * label_mask.unsqueeze(2).expand(-1,-1,self.relation_num)

				#if torch.sum(label_mask) == 0:
				temp_loss = Variable(torch.Tensor([0]).cuda(), requires_grad=True)
				node_num = node_pos.size(0)
				for (h_i,t_j,label) in label_mask:
					#print (predict_re[h_i][t_j],label)
					loss = CE(predict_re[h_i][t_j].unsqueeze(0), torch.LongTensor([label]).cuda())
					temp_loss = temp_loss + loss
				# for h_i in range(node_num):
				# 	for t_j in range(node_num):
				# 		if h_i == t_j:
				# 			continue
				# 		loss = BCE(predict_re[h_i][t_j], label_matrix[h_i][t_j])
				# 		temp_loss = temp_loss + loss
				# for pair in label_mask:
				# 	h_i = pair[0]
				# 	t_j = pair[1]
				# 	loss = BCE(predict_re[h_i,t_j,:], label_matrix[h_i,t_j,:])
				# 	temp_loss = temp_loss + loss
				# temp_loss = temp_loss/len(label_mask)
				temp_loss = temp_loss/(node_num*node_num-node_num)
					
					# loss = BCE(predict_re.view(-1,self.relation_num), label_matrix.view(-1,self.relation_num))
					
					#print (loss)
					#loss1 = BCE_mean(predict_re, label_matrix)
					#print (loss/loss1)
				#else:
				#	loss = BCE(predict_re, label_matrix) / torch.sum(label_mask) / self.relation_num

				total_loss = torch.add(total_loss,temp_loss)

				#loss = torch.sum(BCE(predict_re, label_matrix)*relation_mask.unsqueeze(2)) /  (self.relation_num * torch.sum(relation_mask))
				if global_step % self.batch_size == 0:
					all_loss += total_loss.data.cpu().numpy()
					total_loss = total_loss / self.batch_size
					optimizer.zero_grad()
					total_loss.backward()
					optimizer.step()
					total_loss = Variable(torch.Tensor([0]).cuda(), requires_grad=True)

				# output = torch.argmax(predict_re, dim=-1)
				output = torch.softmax(predict_re,dim = -1).data.cpu().numpy()
				#labels = label_matrix.data.cpu().numpy()

				#relation_label = relation_label.data.cpu().numpy()
				right = 0
				for (h_i,t_j,label) in label_mask:
					r = output[h_i,t_j].argmax()
					if r == label:
						right += 1
				ACC = right/len(label_mask)
				global_step += 1
			
			#print (path_counter)


			#logging('-' * 69)
			elapsed = time.time() - start_time
			print (self.acc_NA.correct,self.acc_NA.total)
			print (self.acc_not_NA.correct,self.acc_not_NA.total)
			logging('| epoch {:2d} | ms/b {:5.2f} | train loss {:5.3f} | acc: {:4.2f} '.format(epoch, elapsed * 1000 / len(self.train_file), float(all_loss), ACC))
			# total_loss = 0
			model.eval()
			#bert.eval()
			P,R,F = self.test(model, model_name)
			model.train()
			#bert.train()
			#logging('| epoch {:3d} | time: {:5.2f}s'.format(epoch, time.time() - eval_start_time))
			start_time = time.time()
			#logging('-' * 69)


			if F > best_f1:
				best_f1 = F
				#best_auc = auc
				best_epoch = epoch
				path = os.path.join(self.checkpoint_dir, model_name)
				torch.save(ori_model.state_dict(), path)
			print('当前epoch：',epoch)
			print('当前F1：',F)
			print('best epoch:',best_epoch)
			print('best_f1',best_f1)

			#del pr_x,pr_y
			print ('garbage',gc.collect())
			# if self.learn_rate * (0.9 ** (epoch/2)) > 1e-6:
			# 	StepLR.step()

		print("Finish training")
		print("Best epoch = %d | f1 = %f" % (best_epoch, best_f1))
		print("Storing best result...")
		print("Finish storing")

	def test(self, model, model_name, output=False, input_theta=-1):
		data_idx = 0
		eval_start_time = time.time()
		# test_result_ignore = []
		total_recall_ignore = 0

		test_result = []
		total_recall = 0
		total_correct = 0
		na_recall = 0
		na_correct = 0
		top1_acc = have_label = 0
		index = 0

		tp = 0
		#all_right = 1066
		all_pre = 0
		count_for_origin_positive =0

		#path_counter = Counter()

		def logging(s, print_=True, log_=True):
			if print_:
				print(s)
			if log_:
				with open(os.path.join(os.path.join("log", model_name)), 'a+') as f_log:
					f_log.write(s + '\n')

		for data in tqdm(self.test_file):   #一个batch的数据
			
			data = self.from_list_to_tensor(data,False)
			#path_counter.update(counter)
			#continue

			document = data['document'].cuda()
			#document_char = data['document_char'].cuda()
			document_ner = data['document_ner'].cuda()
			document_pos = data['document_pos'].cuda()
			#title = data['title'].cuda()
			#title_char = data['title_char'].cuda()
			adj_matrix = data['adj_matrix'].cuda()
			sen_matrix = data['sen_matrix'].cuda()
			pos_matrix_h = data['pos_matrix_h'].cuda()
			pos_matrix_t = data['pos_matrix_t'].cuda()
			node_pos = data['node_pos'].cuda()   #节点特征 最重要
			node_type = data['node_type'].cuda()
			#label_mask = data['label_mask']
			#title = data['title']
			label_mask = data['label_mask']
			node_relative_pos = data['node_relative_pos'].cuda()
			with torch.no_grad():
				#predict_re = model(document,document_char,document_ner,document_pos,title,title_char,adj_matrix,sen_matrix,pos_matrix_h,pos_matrix_t,node_pos)
				#document,_ = bert(document.unsqueeze(0),output_all_encoded_layers = False)
				predict_re = model(document,document_ner,document_pos,adj_matrix,sen_matrix,pos_matrix_h,pos_matrix_t,node_pos,node_type,node_relative_pos)

			predict_re = torch.softmax(predict_re,dim = -1)

			predicts = predict_re.data.cpu().numpy()
			#labels = label_matrix.data.cpu().numpy()

			#relation_label = relation_label.data.cpu().numpy()

			for (i,j,label) in label_mask:
				#if label == 1:
				r = np.argmax(predicts[i, j])
				if r == 1:
					all_pre += 1
				if label == 1:
					count_for_origin_positive += 1
					if r == label:
						tp += 1
		# print('count_for_origin_positive',float(count_for_origin_positive))		#过滤之后1459.0（并不正确）
		P = tp/float(all_pre)
		# R = tp/1066.0
		R = tp/float(count_for_origin_positive)
		F = 2*P*R/(P+R+1e-10)

		# print('| test step | time: {:5.2f}'.format((time.time() - eval_start_time)))
		# print ('total_recall', total_recall)

		# test_result.sort(key = lambda x: x[1], reverse=True)
		# max_n = 1000000
		# test_result = test_result[:min(max_n,len(test_result))]

		# pr_x = []
		# pr_y = []
		# correct = 0
		# w = 0

		# if total_recall == 0:
		# 	total_recall = 1  # for test if recall is zero

		# for i, item in enumerate(test_result):
		# 	correct += item[0]
		# 	pr_y.append(float(correct) / (i + 1))
		# 	pr_x.append(float(correct) / total_recall)
		# 	if item[1] > input_theta:
		# 		w = i

		# pr_x = np.asarray(pr_x, dtype='float32')
		# pr_y = np.asarray(pr_y, dtype='float32')
		# f1_arr = (2 * pr_x * pr_y / (pr_x + pr_y + 1e-20))
		# f1 = f1_arr.max()
		# f1_pos = f1_arr.argmax()
		# p = pr_x[f1_pos]
		# r = pr_y[f1_pos]
		# theta = test_result[f1_pos][1]

		# if input_theta==-1:
		# 	w = f1_pos
		# 	input_theta = theta

		# auc = sklearn.metrics.auc(x = pr_x, y = pr_y)
		#if not self.is_test:
		logging('ALL  : P {:3.4f} | R {:3.4f} | F1 {:3.4f} '.format(P, R, F))
		#else:
		#	logging('ma_f1 {:3.4f} | input_theta {:3.4f} test_result F1 {:3.4f} | AUC {:3.4f}'.format(f1, input_theta, f1_arr[w], auc))

		# if output:
		# 	# output = [x[-4:] for x in test_result[:w+1]]
		# 	output = [{'index': x[2], 'h_idx': x[3], 't_idx': x[4], 'r_idx': x[5], 'r': self.id2rel[x[5]], 'title': self.test_file[x[2]]['title']} for x in test_result[:w+1]]
		# 	json.dump(output, open(os.path.join(self.result_dir, model_name + self.test_prefix + "_index.json"), "w"))

		# plt.plot(pr_x, pr_y, lw=2, label=model_name)
		# plt.legend(loc="upper right")
		# if not os.path.exists(self.fig_result_dir):
		# 	os.mkdir(self.fig_result_dir)
		# plt.savefig(os.path.join(self.fig_result_dir, model_name)

		# pr_x = []
		# pr_y = []
		# correct = correct_in_train = 0
		# w = 0
		# for i, item in enumerate(test_result):
		# 	correct += item[0]
		# 	if item[0] & item[2]:
		# 		correct_in_train += 1
		# 	if correct_in_train==correct:
		# 		p = 0
		# 	else:
		# 		p = float(correct - correct_in_train) / (i + 1 - correct_in_train)
		# 	pr_y.append(p)
		# 	pr_x.append(float(correct) / total_recall)
		# 	if item[1] > input_theta:
		# 		w = i

		# pr_x = np.asarray(pr_x, dtype='float32')
		# pr_y = np.asarray(pr_y, dtype='float32')
		# f1_arr = (2 * pr_x * pr_y / (pr_x + pr_y + 1e-20))
		# f1 = f1_arr.max()

		# auc = sklearn.metrics.auc(x = pr_x, y = pr_y)

		# logging('Ignore ma_f1 {:3.4f} | input_theta {:3.4f} test_result F1 {:3.4f} | AUC {:3.4f}'.format(f1, input_theta, f1_arr[w], auc))

		# del test_result

		return P,R,F #, auc, pr_x, pr_y



	def testall(self, model_pattern, model_name, input_theta):#, ignore_input_theta):
		# 0.2782
		model = model_pattern(config = self)

		model.load_state_dict(torch.load(os.path.join(self.checkpoint_dir, model_name)))
		model.cuda()
		model.eval()
		f1, auc, pr_x, pr_y = self.test(model, model_name, True, input_theta)
