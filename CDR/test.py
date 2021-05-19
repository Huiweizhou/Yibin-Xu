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

# sys.excepthook = IPython.core.ultratb.FormattedTB(mode='Verbose', color_scheme='Linux', call_pdb=1)


parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type = str, default = 'GraphCNN_multihead', help = 'name of the model')
parser.add_argument('--save_name', type = str, default = 'GraphCNN_multihead_checkpoint')

parser.add_argument('--train_prefix', type = str, default = 'dev_train')
parser.add_argument('--test_prefix', type = str, default = 'dev_test')


# parser.add_argument('--model_name', type = str, default = 'BiLSTM', help = 'name of the model')
# parser.add_argument('--save_name', type = str)

# parser.add_argument('--train_prefix', type = str, default = 'train')
# parser.add_argument('--test_prefix', type = str, default = 'dev_dev')
parser.add_argument('--input_theta', type = float, default = -1)
# parser.add_argument('--ignore_input_theta', type = float, default = -1)


args = parser.parse_args()
model = {
	'GraphCNN': models.GraphCNN,
	'GraphCNN_bili': models.GraphCNN_bili,
	'GraphCNN_multihead': models.GraphCNN_multihead,
	'GraphCNN_multihead_bert': models.GraphCNN_multihead_bert,
	'GraphCNN_multihead_bert_gate': models.GraphCNN_multihead_bert_gate,
	'GraphCNN_multihead_bert_gate_cls': models.GraphCNN_multihead_bert_gate_cls,
	'GraphCNN_trans': models.GraphCNN_trans,
	'CNN3': models.CNN3,
	'LSTM': models.LSTM,
	'BiLSTM': models.BiLSTM,
	'ContextAware': models.ContextAware,
}

# AGGAT bert: theta = 0.6618
# bert: theta = 0.3522
# AGGAT bert(after_pretrained): theta = 0.6759
# AGGAT (w/o rnn) bert(after_pretrained): theta = 0.5722
# bert(after_pretrained): theta = 0.6062
# AGGAT (w/o rnn) bert(after_pretrained) curriculum learning low-high: theta = 0.6769
# aftter curriculum low2high pretrained AGGAT (w/o rnn), fine-tuned with curriculum low2high (5e-6, _0): theta = 0.4868
# aftter curriculum low2high pretrained AGGAT (w/o rnn), fine-tuned (5e-6, ): theta = 0.4105

# AGGAT gate bert: theta = 0.5036
# AGGAT gate bert cls: theta = 0.5933

con = config.Config(args)
#con.load_train_data()
con.load_test_data()
# con.set_train_model()
con.testall(model[args.model_name], args.save_name, args.input_theta)#, args.ignore_input_theta)
