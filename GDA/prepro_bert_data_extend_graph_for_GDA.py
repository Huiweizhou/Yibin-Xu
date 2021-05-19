import numpy as np
import os
import re
import random
import json
import pickle
from nltk.tokenize import WordPunctTokenizer
import argparse
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from pytorch_pretrained_bert.tokenization import BertTokenizer
from nltk import word_tokenize,sent_tokenize
import csv

tokenizer = BertTokenizer.from_pretrained('./bio_bert/biobert_v1.1_pubmed', do_lower_case=True)

parser = argparse.ArgumentParser()
parser.add_argument('--in_path', type = str, default =  "../data_GDA/")
parser.add_argument('--out_path', type = str, default = "prepro_data_bert")

args = parser.parse_args()
in_path = args.in_path
out_path = args.out_path
case_sensitive = False

char_limit = 16

# train_text_path = os.path.join(in_path,'training_data/abstracts.txt')
# train_anns_path = os.path.join(in_path,'training_data/anns.txt')
# train_label_path = os.path.join(in_path,'training_data/labels.csv')

# test_text_path = os.path.join(in_path,'testing_data/abstracts.txt')
# test_anns_path = os.path.join(in_path,'testing_data/anns.txt')
# test_label_path = os.path.join(in_path,'testing_data/labels.csv')


ent_repl_dic = {}

pronoun_list = []
with open ('pronoun_list.txt','r') as f:
	for line in f:
		pronoun_list.append(line.strip().lower())

relation_type = {(2, 2): 13535, 
				 (4, 2): 3673,
				 (1, 2): 3146, 
				 (5, 4): 2269, 
				 (4, 1): 1955, 
				 (4, 3): 1848, 
				 (5, 1): 1698, 
				 (4, 4): 1552,
				 (5, 3): 1448, 
				 (5, 2): 1402, 
				 (5, 5): 1367, 
				 (4, 5): 1117, 
				 (1, 1): 860, 
				 (1, 4): 480, 
				 (2, 1): 458, 
				 (2, 4): 391, 
				 (1, 3): 357, 
				 (1, 5): 260,
				 (2, 5): 212, 
				 (2, 3): 152}


def norm_sentence(sentence):
	sentence = sentence + ' '
	sentence = sentence.replace("'s ",' ')
	sentence = sentence.replace("'",' ')
	sentence = sentence.replace('"',' ')

	sentence = sentence.replace(', ',' , ')
	sentence = sentence.replace(': ',' : ')
	sentence = sentence.replace('! ',' ! ')
	sentence = sentence.replace('? ',' ? ')
	sentence = sentence.replace('. ',' . ')
	sentence = sentence.replace('-',' - ')
	
	sentence = re.sub('[\(\)\[\]\{\}]',' ',sentence)
	sentence = re.sub('\b[0-9]+\b','num',sentence)
	sentence = re.sub('\s+',' ',sentence)

	for _ in range (5):
		setence = sentence.replace('  ',' ')
	return sentence.strip().split(' ')		#将句子拆成一个一个的词？

def norm_mentions(sentence,offset,sentence_index):
	sentence = norm_sentence(sentence)
	# print('sentence',sentence)
	new_sentence = []
	mentions = []
	for i, word in enumerate(sentence):		#出现'000000000000000010022756_4_8_G*' 这样的实体，最后跟了*号，需要处理
		if len(word) == 31 and (word[-2] == 'D' or word[-2] == 'G') and word[-1] == '*':
			word_new = word[:-1]
			sentence[i] = word_new
	# print('sentence_new',sentence)


	for word in sentence:
		if len(word) == 30 and (word[-1] == 'D' or word[-1] == 'G'):
			# print('word',word)		#句子里被替换掉的实体mention
			name,ID = ent_repl_dic[word]
			name = norm_sentence(name)
			mentions.append([len(new_sentence) + offset,len(new_sentence) + len(name) + offset,name,word[-1],ID,sentence_index])  # start,end,name,type,ID
			new_sentence.extend(name)
		else:
			new_sentence.append(word)

	return new_sentence,mentions

def normlization_article (texts,mentions):
	sents = sent_tokenize(texts)
	Ls_char = [0]		#这是  保存句子开始位置？？（字符级别）
	Ls = [0]			#这是？  保存句子的开始位置？？（单词级别）
	new_document = []
	# print('mentions ######',mentions)		#传进的参数是有gene的
	mentions = []
	for i,sent in enumerate(sents):
		new_sent,sent_mentions = norm_mentions(sent,Ls[-1],i)
		mentions.extend(sent_mentions)
		new_document.extend(new_sent)
		L = len(new_sent) + Ls[-1]  
		Ls.append(L)
		L = len(sent) + Ls_char[-1] + 1 #because of the  ' '
		Ls_char.append(L)
	# print('debug_mention',mentions)		#此处缺失gene mention
	
	vertexs = {}
	for mention in mentions:
		start = mention[0]
		end = mention[1]
		name = mention[2]
		typee = mention[3]
		meshID = mention[4]
		sentence_index = mention[5]
		if mention[4] not in vertexs:		#如果meshID不在集合中
			vertexs[meshID] = [len(vertexs),[(start,end)],[(Ls[sentence_index],Ls[sentence_index+1])],[name],typee]   # entity_index  #len(vertexs) 用于记录结点编号？
		if mention[4] in vertexs:
			vertexs[meshID][1].append((start,end))
			vertexs[meshID][2].append((Ls[sentence_index],Ls[sentence_index+1]))
			vertexs[meshID][3].append(name)
	return new_document,Ls,vertexs

overlap_count =0
def norm_article(pubID,texts,mentions):
	global overlap_count
	#ori_tit_len = tit_len
	len_e = 30   #fix all the entity to 30 length with uniform IDs
	mentions = sorted(mentions,key = lambda ent: ent[0])
	# print('****************************')
	# print('mentions?????',mentions)
	# print('****************************')
	# if entities == []:
	#     return Article,Article[:tit_len],[],[]
	#print (mentions)
	new_mentions = [mentions[0]]
	ex_word_dic = []
	off_set = 0
	#to remove the overlap entities
	for i in range(1,len(mentions)):
		if mentions[i][0]<new_mentions[-1][1]:		#遍历的mention开始位置在原mention此表第一个元素的结束位置之前？
			if mentions[i][1]-mentions[i][0] > new_mentions[-1][1] - new_mentions[-1][0]:
				new_mentions[-1] = mentions[i]
			else:
				# print (entities[i])
				# print (new_entities[-1])
				overlap_count += 1
		else:
			new_mentions.append(mentions[i])
	# print('new_mentions',new_mentions)		#此处基因type和id被改了？？
	for i in range(len(new_mentions)):
		#if entities[i][0] == '9746003':
		#    print (entities[i])
		# use pubID offset start & end to replace the entity.
		exchange_word = pubID + '_' + str(new_mentions[i][0]) +'_' + str(new_mentions[i][1]) +'_'+ new_mentions[i][3][0]
		exchange_word = '0'*(len_e - len(exchange_word)) + exchange_word  #norm them to be the same length
		# print('exchange_word',exchange_word)	#形如exchange_word 000000000011397889_2111_2132_D
		#ex_word_dic.append(exchange_word)
		if exchange_word == '0001242663_368_393_D':
			print (exchange_word)
		ent_repl_dic [exchange_word] = [new_mentions[i][2], new_mentions[i][4]]   #mention name & id
		len_diff = len(exchange_word) - len(new_mentions[i][2])
		texts = texts[:new_mentions[i][0] + off_set] + exchange_word + texts[ new_mentions[i][1] + off_set:]
		# if new_mentions[i][1] <= ori_tit_len:
		#     tit_len += len_diff
		new_mentions[i][0] += off_set
		new_mentions[i][1] += (off_set + len_diff)
		off_set += len_diff
	return texts,new_mentions#,ex_word_dic

def init(data_pattern, max_length = 512, is_training = True, suffix=''):
	ner2id = {'G':0,'D':1}
	# saving new data
	print("Saving files")
	if is_training:
		name_prefix = "train"
	else:
		name_prefix = "test"

	Ma = 0
	Ma_e = 0
	data = []
	unk_num = 0
	k_num = 0
	i = 0
	#intrain = notintrain = notindevtrain = indevtrain = 0
	storage_size = 10000
	count_for_error =0		#记录多少样例relation列表是空的

	max_exist_sentence_num = 0		#实体最多的mention数
	max_sentence_length = 0		#最长句子长度
	max_coexist_sentence_num = 0		#实体对最多的mention数
	max_coexist_sentence_length = 0		#实体对所在句子最长为

	if data_pattern == 'train':
		pubid_list_path = '../data_GDA/pubid_list_for_train.pkl'
		text_dict_path = '../data_GDA/text_dict_for_train.pkl'
		anns_dict_path = '../data_GDA/anns_dict_for_train.pkl'
		label_dict_path = '../data_GDA/label_dict_for_train.pkl'
	else:
		pubid_list_path = '../data_GDA/pubid_list_for_test.pkl'
		text_dict_path = '../data_GDA/text_dict_for_test.pkl'
		anns_dict_path = '../data_GDA/anns_dict_for_test.pkl'
		label_dict_path = '../data_GDA/label_dict_for_test.pkl'
	
	pubid_list = pickle.load(open(pubid_list_path,'rb'),encoding= 'iso-8859-1')
	text_dict = pickle.load(open(text_dict_path,'rb'),encoding= 'iso-8859-1')
	anns_dict = pickle.load(open(anns_dict_path,'rb'),encoding= 'iso-8859-1')
	label_dict = pickle.load(open(label_dict_path,'rb'),encoding= 'iso-8859-1')
	
	print('*'*15)
	print('len pubid_list',len(pubid_list))      #train 21912  ,test 1000
	print('len text_dict',len(text_dict)) 
	print('len anns_dict',len(anns_dict)) 
	print('len label_dict',len(label_dict)) 
	print('*'*15)

	for i_index in tqdm(range(len(pubid_list))):
	# for i_index in tqdm(range(1)):
		pubID = pubid_list[i_index]
		# pubID = '10190325'		#处理bug的例子
		# pubID = '10215411'
		# pubID = '10022756'
		document = text_dict[pubID]
		# print('document',document)
		mention =[]
		articleGraph = nx.DiGraph(pubID = pubID)
		relation =[]
		label_one = label_dict[pubID]
		for item in label_one:
			if item[0] == pubID:
				geneid = item[1]
				diseaseid = item[2]
				relation.append((geneid,diseaseid))
		# print('relation',relation)
		anns_one = anns_dict[pubID]
		# print('anno_one',anns_one)
		for item in anns_one:
			meshID = item[5].split('|')[0]
			# print('meshID',meshID)
			start_pos = int(item[1])
			end_pos = int(item[2])
			entity_type = item[4]
			mention_name = item[3]
			mention.append([start_pos,end_pos,mention_name,entity_type,meshID])
		# print('mention',mention)

		for i_th in mention:			#解决部分样例中，部分mention同时标注为基因和疾病，此时将疾病标注删除
			for j_th in mention:
				if i_th[0] ==j_th[0] and i_th[1] ==j_th[1]:
					if i_th[-2][0] == 'D' and j_th[-2][0] == 'G':
						if i_th in mention:
							mention.remove(i_th)
					if j_th[-2][0] =='D' and i_th[-2][0] == 'G':
						if j_th in mention:
							mention.remove(j_th)


		# print('mention_new',mention)
		
		document,mention = norm_article (pubID,document,mention)
		# print('mention 2',mention)		#此处基因就丢掉了？？？
		# print('mention_new_2',mention)
		document,Ls,vertex = normlization_article (document,mention)
		# print('vertex',vertex)			#w为什么vertex少结点？？？此处还把gene丢了？？
		# print('document',document)		#document 处理没有问题，分词了。

		for node in vertex:
			node_index = vertex[node][0]
			node_position = vertex[node][1]
			node_sentence = vertex[node][2]
			#node_name = vertex[node][3]
			node_type = vertex[node][4]
			if node_type not in ner2id:
				# print('error')      #经过打印，发现训练集和测试集都不报错
				ner2id[node_type] = len(ner2id)
			articleGraph.add_node(node_index,meshID = node,exist_sentence = node_sentence,exist_pos=node_position,type = node_type)		#对每个实体建立一个节点
		
		###################以上没有问题##########################

		document_bert = ['[CLS]']
		index_id = []
		#print (document)
		for j, word in enumerate(document):
			word = word.lower()
			start = len(document_bert)
			new_word = tokenizer.tokenize(word)
			document_bert.extend(new_word)
			end = len(document_bert)
			index_id.append((start,end))		#记录token级别，句子开始和结束的位置
		#document_bert = document_bert[:511]
		document_bert.append('[SEP]')		#所有句子（文章结尾）加上sep标记
		document_id = tokenizer.convert_tokens_to_ids(document_bert)   

		max_coexist_sentence_length_i = 0
		max_coexist_sentence_num_i =0
		label_mask = []

		item = {}
		item['document'] = document_id
		item['ID'] = pubID 

		for node_j in articleGraph.nodes():		#此处是建立结点之间的边
			for node_k in articleGraph.nodes():
				if node_j != node_k:
					common_sentences = []
					common_sentences_poses = []		#mention 开头和结尾的位置
					common_next_sentences = []
					common_next_sentences_poses = []
					sents_j = articleGraph.nodes[node_j]['exist_sentence']		#结点j（实体），存在的句子，句子开始和结尾的位置，字符级
					poses_j = articleGraph.nodes[node_j]['exist_pos']
					sents_k = articleGraph.nodes[node_k]['exist_sentence']
					poses_k = articleGraph.nodes[node_k]['exist_pos']
					for pos_j,sent_j in zip(poses_j,sents_j):
						for pos_k,sent_k in zip(poses_k,sents_k):			#？？？？不懂
							if sent_j == sent_k:
								common_sentences.append(sent_j)
								common_sentences_poses.append(pos_j+pos_k)
								if sent_j[1]-sent_j[0] > max_coexist_sentence_length:         #最大的句子长度
									max_coexist_sentence_length = sent_j[1]-sent_j[0]
								if sent_j[1]-sent_j[0] > max_coexist_sentence_length_i:         #最大的句子长度，本图中
									max_coexist_sentence_length_i = sent_j[1]-sent_j[0]
								# if len(ori_data[i]['sents'][sent_j]) > max_coexist_sentence_length:         #最大的句子长度
								#  	max_coexist_sentence_length = len(ori_data[i]['sents'][sent_j])
							if sent_j[1] == sent_k[0]:   # sent j is in front of sent k
								flag = False
								for word in document_bert[sent_j[0]:sent_k[1]]:
									if word.lower() in pronoun_list:
										flag = True
										break
								if flag:
									common_next_sentences.append((sent_j[0],sent_k[1]))		#如果前后两个句子中出现此表中的代词，视为一个句子
									common_next_sentences_poses.append(pos_j+pos_k)
							if sent_j[0] == sent_k[1]:   # sent k is in front of sent j
								flag = False
								for word in document_bert[sent_k[0]:sent_j[1]]:
									if word.lower() in pronoun_list:
										flag = True
										break
								if flag:
									common_next_sentences.append((sent_k[0],sent_j[1]))
									common_next_sentences_poses.append(pos_j+pos_k)
								
					if common_sentences != []:
						articleGraph.add_edge(node_j,node_k,sentences = common_sentences,position = common_sentences_poses)
						if len(common_sentences) > max_coexist_sentence_num:              #统计共同出现的句子个数（实际上可能两个实体在同一个句子中多次出现，算多个句子）
							max_coexist_sentence_num = len(common_sentences)
						if len(common_sentences) > max_coexist_sentence_num_i:              #统计共同出现的句子个数（实际上可能两个实体在同一个句子中多次出现，算多个句子，本图中）
							max_coexist_sentence_num_i = len(common_sentences)
					elif common_next_sentences != []:
						articleGraph.add_edge(node_j,node_k,sentences = common_next_sentences,position = common_next_sentences_poses)
						if len(common_next_sentences) > max_coexist_sentence_num:              #统计共同出现的句子个数（实际上可能两个实体在同一个句子中多次出现，算多个句子）
							max_coexist_sentence_num = len(common_next_sentences)
						if len(common_next_sentences) > max_coexist_sentence_num_i:              #统计共同出现的句子个数（实际上可能两个实体在同一个句子中多次出现，算多个句子，本图中）
							max_coexist_sentence_num_i = len(common_next_sentences)

		articleGraph.graph['max_sentence_length'] = max_coexist_sentence_length_i
		articleGraph.graph['max_sentence_num'] = max_coexist_sentence_num_i
		edge_num = len(articleGraph.edges)

		document_length = len(document_id)
		document_pos = np.zeros ((document_length))		#此处一层括号也行
		document_ner = np.zeros ((document_length))
		for meshID in vertex:
			v = vertex[meshID]
			idx = v[0]
			positions = v[1]
			v_type = v[4]
			for vv in positions:
				document_pos[vv[0]:vv[1]] = idx                    #实体位置
				document_ner[vv[0]:vv[1]] = ner2id[v_type]      #实体类别标签
		item['document_pos'] = document_pos
		item['document_ner'] = document_ner

		item['Ls'] = Ls                               #每个句子start位置
		#item['sents'] = ori_data[i]['sents']
		item['graph'] = articleGraph
		relations = []
		for r in relation:		# r		(geneid,diseaseid)
			if r[0] not in vertex:
				continue
			if r[1] not in vertex:
				continue
			geneid = vertex[r[0]][0]
			diseaseid = vertex[r[1]][0]
			relations.append((geneid,diseaseid))
		# print (relations)
		if len(relations)==0:
			# print('有问题的样例pubid：',pubID)
			count_for_error+=1
		# print(articleGraph.nodes(data = True))
		
		item['relation'] = relations
		# print('relatiosn',relations)		#如：  relatiosn [(0, 6), (4, 6), (3, 6)]
		data.append(item)

		Ma = max(Ma, len(vertex))                 #最多有多少个实体
		#Ma_e = max(Ma_e, len(ori_data[i]['labels']))        #最多右多少对关系
		#print (edge_num)
		
		############以下注释是保存代码#################
		if i < 10:
			#print (i)
			nx.draw(articleGraph)
			#print ('total nodes number:', len(data[choose_index]['graph'].nodes))
			plt.savefig('./graph_fig/'+ name_prefix + suffix + '_zeroEdge_'+str(i) +'.jpg')
			plt.show()
			plt.close()
		#print (1,line)
		i+= 1

		if i%storage_size ==0 or i== len(pubid_list):
			pickle.dump(data, open(os.path.join(out_path, name_prefix + suffix + str(int((i-1)/storage_size)) + '.pkl'),'wb'), protocol=pickle.HIGHEST_PROTOCOL)
			data = []
	############以上注释是保存代码#################

	# for i in range (int((len(data)-1)/storage_size) + 1):
	# 	pickle.dump(data[i*storage_size:min((i+1)*storage_size,len(data))], open (os.path.join(out_path, name_prefix + suffix + '_' + str(i) + '.pkl'), "wb"), protocol=pickle.HIGHEST_PROTOCOL)


	print('有问题的样例总数',count_for_error)
			
	print("Finish saving")
	print ('实体最多mention数：%d' %max_exist_sentence_num)
	print ('最长句子长度为：%d' %max_sentence_length)
	print ('实体对最多的mention数：%d' %max_coexist_sentence_num)
	print ('实体对所在句子长度最长为：%d' %max_coexist_sentence_length)




init(data_pattern='train' , max_length =512, is_training = True, suffix = '_train')
# init(data_pattern='test', max_length = 512, is_training =False, suffix = '_test')