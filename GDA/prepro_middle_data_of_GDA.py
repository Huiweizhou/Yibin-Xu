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

train_text_path = os.path.join(in_path,'training_data/abstracts.txt')
train_anns_path = os.path.join(in_path,'training_data/anns.txt')
train_label_path = os.path.join(in_path,'training_data/labels.csv')

test_text_path = os.path.join(in_path,'testing_data/abstracts.txt')
test_anns_path = os.path.join(in_path,'testing_data/anns.txt')
test_label_path = os.path.join(in_path,'testing_data/labels.csv')
# print('test_text_path',test_text_path)
# print('test_anns_path',test_anns_path)
# print('test_label_path',test_label_path)

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
    new_sentence = []
    mentions = []
    for word in sentence:
        if len(word) == 30 and (word[-1] == 'D' or word[-1] == 'G'):		#句子里被替换掉的实体mention
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
    mentions = []
    for i,sent in enumerate(sents):
        new_sent,sent_mentions = norm_mentions(sent,Ls[-1],i)
        mentions.extend(sent_mentions)
        new_document.extend(new_sent)
        L = len(new_sent) + Ls[-1]  
        Ls.append(L)
        L = len(sent) + Ls_char[-1] + 1 #because of the  ' '
        Ls_char.append(L)
    
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
    if data_pattern == 'train':
        data_text_path = train_text_path
        data_anns_path = train_anns_path
        data_label_path = train_label_path
    else:
        data_text_path = test_text_path
        data_anns_path = test_anns_path
        data_label_path = test_label_path
    
    max_exist_sentence_num = 0
    max_sentence_length = 0
    max_coexist_sentence_num = 0
    max_coexist_sentence_length = 0

    ner2id = {'G':0,'D':1}
    # saving new data
    print("Saving files")
    if is_training:
        name_prefix = "train"
    else:
        name_prefix = "test"


    # labels = []			#保存类似CDR语料的标签
    # with open(data_label_path) as f:
    #     labels_csv = csv.reader(f)
    #     for row in labels_csv:
    #         # print(row)
    #         # assert len(row) == 4
    #         labels.append(row)
    # # print('labels',labels)		#各个元素形如	['20089160', '3577', 'D016773', '1']  pubid，基因，疾病，label
    # # print('labels length',len(labels))		#训练集labels个数44842


    # anns = []		#保存类似CDR语料的mention标注
    # with open(data_anns_path,'r') as f:
    #     lines = f.readlines()
    #     # print('len lines',len(lines))		#训练集568359，测试集18928
    #     for line in lines:
    #         line = line.strip().split('\t')
    #         # print(line)
    #         # print('len anns',len(line))		#长度只为1或者6
    #         if len(line) > 1:
    #             anns.append(line)
    #         else:
    #             continue
    # # for item in anns:
    # # 	print(item)		#形如：    ['17028196', '280', '283', 'APC', 'Disease', 'D011125']
    # # print('len(anns)',len(anns))		#训练集539168，测试集17928


    text_dict = {}		#用来存储abstracts抽取出的数据，字典格式，'pubid':text
    pubid_list = []
    with open(data_text_path,'r') as f:
        lines = f.readlines()
        # print('len lines',len(lines))		#训练集116767
        for i in range(len(lines)):
            if i % 4 == 0:
                pubid = lines[i].strip()		#最后有换行？？
                # print(i)
                assert (len(pubid) >1 and len(pubid) < 9)	#有诸如1071603和187612和45830这样不是八位数字的。。
                title = lines[i+1].strip()
                # print(title)
                abstract = lines[i+2].strip()
                # print(abstract)
                if abstract == '':	
                    text = title
                else:
                    text = title + ' ' + abstract
                text_dict[pubid] = text
                pubid_list.append(pubid)
    # print('len text_dict',len(text_dict))		#训练集样例个数 29192，测试集样例个数1000
    # print('len pubid_list',len(pubid_list))

    ##下面用于保存中间文件
    # text_dict_path = '../data_GDA/test_dict_for'+ suffix +'.pkl'
    # pickle.dump(text_dict,open(text_dict_path,'wb'))
    # print('text _dict_path',suffix,'have been load')
    # pubid_list_path = '../data_GDA/pubid_list_for' + suffix + '.pkl'
    # pickle.dump(pubid_list, open(pubid_list_path,'wb'))
    # print('pubid_list for ',suffix,'have been load')

    labels = []			#保存类似CDR语料的标签
    with open(data_label_path) as f:
        labels_csv = csv.reader(f)
        for row in labels_csv:
            # print(row)
            # assert len(row) == 4
            labels.append(row)
    # print('labels',labels)		#各个元素形如	['20089160', '3577', 'D016773', '1']  pubid，基因，疾病，label
    # print('labels length',len(labels))		#训练集labels个数44842
    label_dict = {}
    for pubid in pubid_list:
        label_for_one = []
        for row in labels:
            if pubid == row[0]:
                label_for_one.append(row)
            else:
                continue
        label_dict[pubid] = label_for_one
    ##下面用于第一次存档
    # label_dict_file_path = '../data_GDA/label_dict_for' + suffix + '.pkl'      #../data_GDA/label_dict_for_train.json
    # pickle.dump(label_dict,open(label_dict_file_path,'wb'))
    # print(suffix,'的 label_dict 已经保存')
    

    anns = []		#保存类似CDR语料的mention标注
    with open(data_anns_path,'r') as f:
        lines = f.readlines()
        # print('len lines',len(lines))		#训练集568359，测试集18928
        for line in lines:
            line = line.strip().split('\t')
            # print(line)
            # print('len anns',len(line))		#长度只为1或者6
            if len(line) > 1:
                anns.append(line)
            else:
                continue
    # for item in anns:
    # 	print(item)		#形如：    ['17028196', '280', '283', 'APC', 'Disease', 'D011125']
    # print('len(anns)',len(anns))		#训练集539168，测试集17928
    anns_dict = {}
    for pubid in pubid_list:
        anns_for_one =[]
        for row in anns:
            if pubid == row[0]:
                anns_for_one.append(row)
            else:
                continue
        anns_dict[pubid] = anns_for_one
    ##下面用于第一次存档
    anns_dict_path = '../data_GDA/anns_dict_for' + suffix + '.pkl'
    pickle.dump(anns_dict,open(anns_dict_path,'wb'))
    print('anns_dict for ',suffix,'have been load')



    Ma = 0
    Ma_e = 0
    data = []
    unk_num = 0
    k_num = 0
    i = 0
    #intrain = notintrain = notindevtrain = indevtrain = 0
    storage_size = 10000

    # for pubid in pubid_list:
    for i in tqdm(range(len(pubid_list))):
        pubID = pubid_list[i]
        # pubID = pubid
        document = text_dict[pubID]
        # print('document',document)
        mention = []
        articleGraph = nx.DiGraph(pubID = pubID)
        relation = []
        for item in labels:
            if item[0] == pubID:
                geneid = item[1]
                diseaseid = item[2]
                relation.append((geneid,diseaseid))
        # print(relation)
        for item in anns:
            meshID = item[5].split('|')[0]
            # print('meshID',meshID)
            start_pos = int(item[1])
            end_pos = int(item[2])
            entity_type = item[4]
            mention_name = item[3]
            mention.append([start_pos,end_pos,mention_name,entity_type,meshID])
        
        document,mention = norm_article (pubID,document,mention)
        document,Ls,vertex = normlization_article (document,mention)







            





# init(data_pattern='train' , max_length =512, is_training = False, suffix = '_train')
init(data_pattern='test', max_length = 512, is_training ='False', suffix = '_test')