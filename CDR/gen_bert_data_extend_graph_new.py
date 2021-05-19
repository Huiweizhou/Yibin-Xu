import numpy as np
import os
import sys
import re
import pickle
# from nltk.tokenize import WordPunctTokenizer
import argparse
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# from tqdm import tqdm
from pytorch_pretrained_bert.tokenization import BertTokenizer
from nltk import word_tokenize, sent_tokenize

os.chdir(sys.path[0])

# from sklearn.externals import joblib   # 解决pickle不能存储大数据的问题
tokenizer = BertTokenizer.from_pretrained('./bio_bert/biobert_v1.1_pubmed',
                                          do_lower_case=True)

parser = argparse.ArgumentParser()
parser.add_argument('--in_path',
                    type=str,
                    default="../data_CDR/original_data/")
parser.add_argument('--out_path', type=str, default="prepro_data_bert")

# rank_path = './result/GraphCNN_multihead_bert_afterpretrain_hop2_rank_checkpointtrain_rank.pkl'
# rank_list = pickle.load(open(rank_path,'rb'))

args = parser.parse_args()
in_path = args.in_path
out_path = args.out_path
case_sensitive = False

char_limit = 16
# train_distant_file_name = os.path.join(in_path, 'CDR_TrainingSet_distance.PubTator.txt')
train_annotated_file_name = os.path.join(in_path,
                                         'CDR_TrainingSet.PubTator.txt')
dev_file_name = os.path.join(in_path, 'CDR_DevelopmentSet.PubTator.txt')
test_file_name = os.path.join(in_path, 'CDR_TestSet.PubTator.txt')

ent_repl_dic = {}

pronoun_list = []
with open('pronoun_list.txt', 'r') as f:
    for line in f:
        pronoun_list.append(line.strip().lower())

relation_type = {
    (2, 2): 13535,
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
    (2, 3): 152
}


def norm_sentence(sentence):
    sentence = sentence + ' '
    sentence = sentence.replace("'s ", ' ')
    sentence = sentence.replace("'", ' ')
    sentence = sentence.replace('"', ' ')

    sentence = sentence.replace(', ', ' , ')
    sentence = sentence.replace(': ', ' : ')
    sentence = sentence.replace('! ', ' ! ')
    sentence = sentence.replace('? ', ' ? ')
    sentence = sentence.replace('. ', ' . ')
    sentence = sentence.replace('-', ' - ')

    sentence = re.sub('[\(\)\[\]\{\}]', ' ', sentence)
    sentence = re.sub('\b[0-9]+\b', 'num', sentence)
    sentence = re.sub('\s+', ' ', sentence)

    for _ in range(5):
        setence = sentence.replace('  ', ' ')
    return sentence.strip().split(' ')


def norm_mentions(sentence, offset, sentence_index):
    sentence = norm_sentence(sentence)
    new_sentence = []
    mentions = []
    # print(sentence)
    for word in sentence:
        if len(word) == 30 and (word[-1] == 'C' or word[-1] == 'D'):
            name, ID = ent_repl_dic[word]
            name = norm_sentence(name)
            mentions.append([
                len(new_sentence) + offset,
                len(new_sentence) + len(name) + offset, name, word[-1], ID,
                sentence_index
            ])
            # start,end,name,type,ID
            new_sentence.extend(name)
        else:
            new_sentence.append(word)

    return new_sentence, mentions


def normlization_article(texts, mentions):
    sents = sent_tokenize(texts)
    Ls_char = [0]
    Ls = [0]
    new_document = []
    mentions = []
    re_sents = []
    for i, sent in enumerate(sents):
        # print(sents)
        new_sent, sent_mentions = norm_mentions(sent, Ls[-1], i)  # 词，句中的实体信息
        # print(new_sent)
        # print(sent_mentions)
        mentions.extend(sent_mentions)
        new_document.extend(new_sent)
        re_sents.append(new_sent)
        L = len(new_sent) + Ls[-1]
        Ls.append(L)  # 按单词划分，句子长度
        L = len(sent) + Ls_char[-1] + 1  # because of the  ' '
        Ls_char.append(L)  # 按字符划分，句子长度

    vertexs = {}
    for mention in mentions:
        start = mention[0]
        end = mention[1]
        name = mention[2]
        typee = mention[3]
        meshID = mention[4]
        sentence_index = mention[5]
        if mention[4] not in vertexs:
            vertexs[meshID] = [
                len(vertexs), [(start, end)],
                [(Ls[sentence_index], Ls[sentence_index + 1])], [name], typee
            ]  # entity_index
        if mention[4] in vertexs:
            if (start, end) in vertexs[meshID][1]:
                continue
            else:
                vertexs[meshID][1].append((start, end))
                vertexs[meshID][2].append(
                    (Ls[sentence_index], Ls[sentence_index + 1]))
                vertexs[meshID][3].append(name)


# mentions 带句子标志，第几个
    return new_document, Ls, vertexs, mentions, re_sents


def norm_article(pubID, texts, mentions):
    global overlap_count
    # ori_tit_len = tit_len
    len_e = 30  # fix all the entity to 30 length with uniform IDs
    mentions = sorted(mentions, key=lambda ent: ent[0])

    new_mentions = [mentions[0]]
    # ex_word_dic = []
    off_set = 0
    # to remove the overlap entities
    for i in range(1, len(mentions)):
        if mentions[i][0] < new_mentions[-1][1]:
            if mentions[i][1] - mentions[i][0] > new_mentions[-1][
                    1] - new_mentions[-1][0]:
                new_mentions[-1] = mentions[i]
            else:
                overlap_count += 1
        else:
            new_mentions.append(mentions[i])
    for i in range(len(new_mentions)):
        # if entities[i][0] == '9746003':
        #     print (entities[i])
        #  use pubID offset start & end to replace the entity.
        exchange_word = pubID + '_' + str(new_mentions[i][0]) + '_' + str(
            new_mentions[i][1]) + '_' + new_mentions[i][3][0]
        exchange_word = '0' * (len_e - len(
            exchange_word)) + exchange_word  # norm them to be the same length
        # ex_word_dic.append(exchange_word)
        if exchange_word == '0001242663_368_393_D':
            print(exchange_word)
        ent_repl_dic[exchange_word] = [new_mentions[i][2],
                                       new_mentions[i][4]]  # mention name & id
        len_diff = len(exchange_word) - len(new_mentions[i][2])
        texts = texts[:new_mentions[i][0] +
                      off_set] + exchange_word + texts[new_mentions[i][1] +
                                                       off_set:]
        #  if new_mentions[i][1] <= ori_tit_len:
        #      tit_len += len_diff
        new_mentions[i][0] += off_set
        new_mentions[i][1] += (off_set + len_diff)
        # print(texts)
        off_set += len_diff
    return texts, new_mentions


def KMP(subtext, text):  # 改进的KMP算法，找出串出现的所有位置
    def get_next(subtext):
        a = i = p = 0
        m = len(subtext)
        next = [0] * m
        next[0] = m
        for i in range(1, m - 1):
            if (i >= p) or (i + next[i - a] >= p):
                if i >= p:
                    p = i
                    while (p < m) and (subtext[p] == subtext[p - i]):
                        p += 1
                    next[i] = p - i
                    a = i
                else:
                    next[i] = next[i - a] - 1
        return next

    a = i = p = 0
    m = len(subtext)
    next = get_next(subtext)
    n = len(text)
    extend = [0] * n
    result = []
    for i in range(1, n - 1):
        if (i >= p) or (i + next[i - a] >= p): # i >= p 的作用：举个典型例子，text 和 subtext 无一字符相同
            if i >= p:
                p = i
            while (p < n) and (p - i < m) and (text[p] == subtext[p - i]):
                p += 1
            extend[i] = p - i
            if extend[i] == m:
                result.append(i)
            a = i
        else:
            extend[i] = next[i - a]
            if extend[i] == m:
                result.append(i)
    return result


def update_sent_pos(mentions, vertex, sents):
    # print(mentions)
    sent_pos = [0]
    length = 0
    start_end_pos = {}
    bert_sent = ['CLS']
    name_meshID = {}
    for m in mentions:
        meshID = m[4]
        name = m[2]
        name_meshID[meshID] = name
        start_end_pos[meshID] = []
    for i, sent in enumerate(sents):
        # length = 0  # 累加切分后的词数
        # pre_word = ' '  # 前一个词
        for j, word in enumerate(sent):
            # tag = 0
            new_word = tokenizer.tokenize(word.lower())
            bert_sent += new_word
        if i == len(sents) - 1:
            bert_sent += ['SEP']
        sent_pos.append(length + len(bert_sent))
    # print(bert_sent)
    for mid in name_meshID:
        temp_bert_sent = []
        # print(name_meshID[mid])
        for word in name_meshID[mid]:
            temp_bert_sent += tokenizer.tokenize(word.lower())
        start_list = KMP(temp_bert_sent, bert_sent)
        for i in start_list:
            if i == 0:
                start_end_pos[mid].append((i, i + len(temp_bert_sent)))
            elif bert_sent[i - 1] != '-':
                start_end_pos[mid].append((i, i + len(temp_bert_sent)))

    meshID_sent_pos = {}
    for mention in mentions:
        meshID = mention[4]
        sentence_index = mention[5]
        if mention[4] not in meshID_sent_pos:
            meshID_sent_pos[meshID] = [(sent_pos[sentence_index],
                                        sent_pos[sentence_index + 1])]
        else:
            meshID_sent_pos[meshID].append(
                (sent_pos[sentence_index], sent_pos[sentence_index + 1]))
    for node in vertex:
        vertex[node][2] = meshID_sent_pos[node]
        vertex[node][1] = start_end_pos[node]
    return vertex


def init(data_file_name, max_length=512, is_training=True, suffix=''):

    max_exist_sentence_num = 0
    max_sentence_length = 0
    max_coexist_sentence_num = 0
    max_coexist_sentence_length = 0

    ner2id = {'C': 0, 'D': 1}

    #  saving new data
    print("Saving files")
    if is_training:
        name_prefix = "train"
    else:
        name_prefix = "dev"

    Ma = 0
    # Ma_e = 0
    data = []
    # unk_num = 0
    # k_num = 0
    i = 0
    # intrain = notintrain = notindevtrain = indevtrain = 0
    storage_size = 10000

    with open(data_file_name, 'r') as f:
        line = f.readline()
        while line:
            line = line.strip().split('|')
            pubID = line[0]
            title = line[2]
            abstract = f.readline().strip().split('|')[2]
            # print (abstract)
            document = title + ' ' + abstract
            # vertex = {}
            mention = []
            articleGraph = nx.DiGraph(pubID=pubID)
            relation = []
            while True:
                line = f.readline().strip().split('\t')
                # print (line)
                if len(line) == 1:
                    break
                elif line[1] == 'CID':
                    chemical = line[2]
                    disease = line[3]

                    relation.append((chemical, disease))
                elif len(line) > 4 and (line[4][0] == 'C' or line[4][0]
                                        == 'D'):  # entity mention
                    meshID = line[5].split('|')[0]
                    if meshID == '-1':
                        continue
                    start_pos = int(line[1])
                    end_pos = int(line[2])
                    mention_name = line[3]
                    entity_type = line[4]
                    mention.append([
                        start_pos, end_pos, mention_name, entity_type, meshID
                    ])

            document, mention = norm_article(pubID, document, mention)
            document, Ls, vertex, mentions, sents = normlization_article(
                document, mention)
            document_bert = ['[CLS]']
            index_id = []
            for j, word in enumerate(document):
                word = word.lower()
                start = len(document_bert)
                new_word = tokenizer.tokenize(word)
                # length += len(new_word)
                document_bert.extend(new_word)
                end = len(document_bert)
                index_id.append((start, end))
            document_bert.append('[SEP]')
            document_id = tokenizer.convert_tokens_to_ids(document_bert)
            vertex = update_sent_pos(mentions, vertex, sents)
            for node in vertex:
                node_index = vertex[node][0]
                node_position = vertex[node][1]
                node_sentence = vertex[node][2]
                # node_name = vertex[node][3]
                node_type = vertex[node][4]
                if node_type not in ner2id:
                    ner2id[node_type] = len(ner2id)
                articleGraph.add_node(node_index,
                                      meshID=node,
                                      exist_sentence=node_sentence,
                                      exist_pos=node_position,
                                      type=node_type)
            max_coexist_sentence_length_i = 0
            max_coexist_sentence_num_i = 0
            # label_mask = []
            item = {}
            item['document'] = document_id
            item['ID'] = pubID

            for node_j in articleGraph.nodes():
                for node_k in articleGraph.nodes():
                    if node_j != node_k:
                        common_sentences = []
                        common_sentences_poses = []
                        common_next_sentences = []
                        common_next_sentences_poses = []
                        sents_j = articleGraph.nodes[node_j]['exist_sentence']
                        poses_j = articleGraph.nodes[node_j]['exist_pos']
                        sents_k = articleGraph.nodes[node_k]['exist_sentence']
                        poses_k = articleGraph.nodes[node_k]['exist_pos']
                        for pos_j, sent_j in zip(poses_j, sents_j):
                            for pos_k, sent_k in zip(poses_k, sents_k):
                                if sent_j == sent_k:
                                    common_sentences.append(sent_j)
                                    common_sentences_poses.append(pos_j +
                                                                  pos_k)
                                    if sent_j[1] - sent_j[
                                            0] > max_coexist_sentence_length:  # 最大的句子长度
                                        max_coexist_sentence_length = sent_j[
                                            1] - sent_j[0]
                                    if sent_j[1] - sent_j[
                                            0] > max_coexist_sentence_length_i:  # 最大的句子长度，本图中
                                        max_coexist_sentence_length_i = sent_j[
                                            1] - sent_j[0]
                                    #  if len(ori_data[i]['sents'][sent_j]) > max_coexist_sentence_length:         # 最大的句子长度
                                    #   	max_coexist_sentence_length = len(ori_data[i]['sents'][sent_j])
                                if sent_j[1] == sent_k[
                                        0]:  # sent j is in front of sent k
                                    flag = False
                                    for word in document_bert[
                                            sent_j[0]:sent_k[1]]:
                                        if word.lower() in pronoun_list:
                                            flag = True
                                            break
                                    if flag:
                                        common_next_sentences.append(
                                            (sent_j[0], sent_k[1]))
                                        common_next_sentences_poses.append(
                                            pos_j + pos_k)
                                if sent_j[0] == sent_k[
                                        1]:  # sent j is in front of sent k
                                    flag = False
                                    for word in document_bert[
                                            sent_k[0]:sent_j[1]]:
                                        if word.lower() in pronoun_list:
                                            flag = True
                                            break
                                    if flag:
                                        common_next_sentences.append(
                                            (sent_k[0], sent_j[1]))
                                        common_next_sentences_poses.append(
                                            pos_j + pos_k)

                        if common_sentences != []:
                            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
                            articleGraph.add_edge(
                                node_j,
                                node_k,
                                sentences=common_sentences,
                                position=common_sentences_poses)
                            if len(
                                    common_sentences
                            ) > max_coexist_sentence_num:  # 统计共同出现的句子个数（实际上可能两个实体在同一个句子中多次出现，算多个句子）
                                max_coexist_sentence_num = len(
                                    common_sentences)
                            if len(
                                    common_sentences
                            ) > max_coexist_sentence_num_i:  # 统计共同出现的句子个数（实际上可能两个实体在同一个句子中多次出现，算多个句子，本图中）
                                max_coexist_sentence_num_i = len(
                                    common_sentences)
                        elif common_next_sentences != []:
                            articleGraph.add_edge(
                                node_j,
                                node_k,
                                sentences=common_next_sentences,
                                position=common_next_sentences_poses)
                            if len(
                                    common_next_sentences
                            ) > max_coexist_sentence_num:  # 统计共同出现的句子个数（实际上可能两个实体在同一个句子中多次出现，算多个句子）
                                max_coexist_sentence_num = len(
                                    common_next_sentences)
                            if len(
                                    common_next_sentences
                            ) > max_coexist_sentence_num_i:  # 统计共同出现的句子个数（实际上可能两个实体在同一个句子中多次出现，算多个句子，本图中）
                                max_coexist_sentence_num_i = len(
                                    common_next_sentences)

            articleGraph.graph[
                'max_sentence_length'] = max_coexist_sentence_length_i
            articleGraph.graph['max_sentence_num'] = max_coexist_sentence_num_i
            edge_num = len(articleGraph.edges)

            document_length = len(document_id)
            document_pos = np.zeros((document_length))
            document_ner = np.zeros((document_length))
            for meshID in vertex:
                v = vertex[meshID]
                idx = v[0]
                positions = v[1]
                v_type = v[4]
                for vv in positions:
                    document_pos[vv[0]:vv[1]] = idx  # 实体位置
                    document_ner[vv[0]:vv[1]] = ner2id[v_type]  # 实体类别标签
            item['document_pos'] = document_pos
            item['document_ner'] = document_ner

            item['Ls'] = Ls  # 每个句子start位置
            # item['sents'] = ori_data[i]['sents']
            item['graph'] = articleGraph
            relations = []
            for r in relation:
                if r[0] not in vertex:
                    continue
                if r[1] not in vertex:
                    continue
                chem = vertex[r[0]][0]
                dis = vertex[r[1]][0]
                relations.append((chem, dis))
            # print (relations)
            item['relation'] = relations
            data.append(item)

            Ma = max(Ma, len(vertex))  # 最多有多少个实体
            # Ma_e = max(Ma_e, len(ori_data[i]['labels']))        # 最多右多少对关系
            # print (edge_num)
            if i == 0:
                # print (i)
                nx.draw(articleGraph)
                # print ('total nodes number:', len(data[choose_index]['graph'].nodes))
                plt.savefig('./graph_fig/' + name_prefix + suffix +
                            '_zeroEdge_' + str(i) + '.jpg')
                plt.show()
                plt.close()
            line = f.readline()
            # print (1,line)
            i += 1
        for i in range(int((len(data) - 1) / storage_size) + 1):
            pickle.dump(
                data[i * storage_size:min((i + 1) * storage_size, len(data))],
                open(
                    os.path.join(out_path,
                                 name_prefix + suffix + '_' + str(i) + '.pkl'),
                    "wb"),
                protocol=pickle.HIGHEST_PROTOCOL)

    print("Finish saving")
    print('实体最多mention数：%d' % max_exist_sentence_num)
    print('最长句子长度为：%d' % max_sentence_length)
    print('实体对最多的mention数：%d' % max_coexist_sentence_num)
    print('实体对所在句子长度最长为：%d' % max_coexist_sentence_length)


init(train_annotated_file_name,
     max_length=512,
     is_training=False,
     suffix='_train')
init(dev_file_name, max_length=512, is_training=False, suffix='_dev')
init(test_file_name, max_length=512, is_training=False, suffix='_test')
