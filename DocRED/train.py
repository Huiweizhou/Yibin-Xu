import config
import models
import numpy as np
import os
import time
import datetime
import json
from sklearn.metrics import average_precision_score
import sys
import os
import argparse
# import IPython



parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type = str, default = 'GraphCNN_multihead_bert_gate_cls', help = 'name of the model')
parser.add_argument('--save_name', type = str, default = 'GraphCNN_multihead_bert_gate_cls_checkpoint')

parser.add_argument('--train_prefix', type = str, default = 'dev_train')
parser.add_argument('--test_prefix', type = str, default = 'dev_dev')
  

args = parser.parse_args()
model = {
	'GraphCNN_multihead_gate' : models.GraphCNN_multihead_gate,
	'GraphCNN_multihead_bert_gate_cls': models.GraphCNN_multihead_bert_gate_cls,
	'CNN3': models.CNN3,
	'LSTM': models.LSTM,
	'BiLSTM': models.BiLSTM,
	'ContextAware': models.ContextAware,
}

con = config.Config(args)
con.set_max_epoch(200)
con.load_train_data()
con.load_test_data()
# con.gen_train_facts_anno()
con.gen_train_facts_distant()
# con.set_train_model()
con.train(model[args.model_name], args.save_name)
