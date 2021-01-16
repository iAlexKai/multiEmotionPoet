"""
Copyright 2018 NAVER Corp.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
   and IDIAP Research Institute nor the names of its contributors may be
   used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""
import time
import math
import warnings
import tqdm
import torch
import tensorflow as tf
import numpy as np

warnings.filterwarnings("ignore")
config = tf.compat.v1.ConfigProto()
config.intra_op_parallelism_threads = 12
config.inter_op_parallelism_threads = 2
tf.compat.v1.Session(config=config)
# os.environ['CUDA_VISIBLE_DEVICES'] = '/gpu:0'


def asHHMMSS(s):
    m = math.floor(s / 60)
    s -= m * 60
    h = math.floor(m /60)
    m -= h *60
    return '%d:%d:%d'% (h, m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s<%s'%(asHHMMSS(s), asHHMMSS(rs))


#######################################################################
def sent2indexes(sentence, vocab):
    def convert_sent(sent, vocab):
        return np.array([vocab[word] for word in sent.split(' ')])
    if type(sentence) is list:
        indexes=[convert_sent(sent, vocab) for sent in sentence]
        sent_lens = [len(idxes) for idxes in indexes]
        max_len = max(sent_lens)
        inds = np.zeros((len(sentence), max_len), dtype=np.int)
        for i, idxes in enumerate(indexes):
            inds[i,:len(idxes)]=indexes[i]
        return inds
    else:
        return convert_sent(sentence, vocab)


def indexes2sent(indexes, vocab, eos_tok, ignore_tok=0): 
    '''indexes: numpy array'''
    def revert_sent(indexes, ivocab, eos_tok, ignore_tok=0):
        toks=[]
        length=0
        indexes=filter(lambda i: i!=ignore_tok, indexes)
        for idx in indexes:
            toks.append(ivocab[idx])
            length+=1
            if idx == eos_tok:
                break
        return ' '.join(toks), length
    
    ivocab = {v: k for k, v in vocab.items()}
    if indexes.ndim==1:# one sentence
        return revert_sent(indexes, ivocab, eos_tok, ignore_tok)
    else:# dim>1
        sentences=[] # a batch of sentences
        lens=[]
        for inds in indexes:
            sentence, length = revert_sent(inds, ivocab, eos_tok, ignore_tok)
            sentences.append(sentence)
            lens.append(length)
        return sentences, lens



from torch.nn import functional as F

use_cuda = torch.cuda.is_available()


def to_tensor(data):
    tensor = data
    if isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data)
    if use_cuda:
        tensor = tensor.cuda()
    return tensor


def gaussian_kld(recog_mu, recog_logvar, prior_mu, prior_logvar):
    kld = -0.5 * torch.sum(1 + (recog_logvar - prior_logvar)
                               - torch.div(torch.pow(prior_mu - recog_mu, 2), torch.exp(prior_logvar))
                               - torch.div(torch.exp(recog_logvar), torch.exp(prior_logvar)), 1)
    return kld


SENTENCE_LIMIT_SIZE = 7
with open('data/vocab.txt') as vocab_file:
    vocab = vocab_file.read().strip().split('\n')
    # import pdb
    # pdb.set_trace()


# ### 构造映射

# In[30]:

# 单词到编码的映射，例如machine -> 10283
word_to_token = {word: token for token, word in enumerate(vocab)}
# 编码到单词的映射，例如10283 -> machine
token_to_word = {token: word for word, token in word_to_token.items()}

# ### 转换文本

# In[61]:

def convert_text_to_token(sentence, word_to_token_map=word_to_token, limit_size=SENTENCE_LIMIT_SIZE):
    """
    根据单词-编码映射表将单个句子转化为token

    @param sentence: 句子，str类型
    @param word_to_token_map: 单词到编码的映射
    @param limit_size: 句子最大长度。超过该长度的句子进行截断，不足的句子进行pad补全

    return: 句子转换为token后的列表
    """
    # 获取unknown单词和pad的token
    unk_id = word_to_token_map["<unk>"]
    pad_id = word_to_token_map["<pad>"]

    # 对句子进行token转换，对于未在词典中出现过的词用unk的token填充
    tokens = [word_to_token_map.get(word, unk_id) for word in sentence.lower()]

    # Pad
    if len(tokens) < limit_size:
        tokens.extend([0] * (limit_size - len(tokens)))
    # Trunc
    else:
        tokens = tokens[:limit_size]

    return tokens

word_to_vec = {}

# ### 构造词向量矩阵

# In[91]:

VOCAB_SIZE = len(vocab)  # 10384
EMBEDDING_SIZE = 300

# In[92]:

# 初始化词向量矩阵（这里命名为static是因为这个词向量矩阵用预训练好的填充，无需重新训练）
static_embeddings = np.zeros([VOCAB_SIZE, EMBEDDING_SIZE])

for word, token in tqdm.tqdm(word_to_token.items()):
    # 用glove词向量填充，如果没有对应的词向量，则用随机数填充
    word_vector = word_to_vec.get(word, 0.2 * np.random.random(EMBEDDING_SIZE) - 0.1)
    static_embeddings[token, :] = word_vector

# 重置PAD为0向量
pad_id = word_to_token["<pad>"]
static_embeddings[pad_id, :] = np.zeros(EMBEDDING_SIZE)

# In[93]:

static_embeddings = static_embeddings.astype(np.float32)

# 清空图
tf.compat.v1.reset_default_graph()

# In[31]:

# 定义神经网络超参数
HIDDEN_SIZE = 512
LEARNING_RATE = 0.001
EPOCHES = 50
BATCH_SIZE = 256

# In[32]:
# DNN:

model_name = 'dnn'

with tf.name_scope("dnn"):
    # 输入及输出tensor
    with tf.name_scope("placeholders"):
        inputs = tf.compat.v1.placeholder(dtype=tf.int32, shape=(None, 7), name="inputs")
        targets = tf.compat.v1.placeholder(dtype=tf.int64, shape=(None), name="targets")

    # embeddings
    with tf.name_scope("embeddings"):
        # 用pre-trained词向量来作为embedding层
        embedding_matrix = tf.Variable(initial_value=static_embeddings, trainable=False, name="embedding_matrix")
        embed = tf.nn.embedding_lookup(embedding_matrix, inputs, name="embed")
        # 相加词向量得到句子向量
        sum_embed = tf.reduce_sum(embed, axis=1, name="sum_embed")

    # model
    with tf.name_scope("model"):
        # 隐层权重
        W1 = tf.Variable(tf.compat.v1.random_normal(shape=(EMBEDDING_SIZE, HIDDEN_SIZE), stddev=0.1), name="W1")
        b1 = tf.Variable(tf.zeros(shape=(HIDDEN_SIZE), name="b1"))

        # 输出层权重
        W2 = tf.Variable(tf.compat.v1.random_normal(shape=(HIDDEN_SIZE, 3), stddev=0.1), name="W2")
        b2 = tf.Variable(tf.zeros(shape=(1), name="b2"))

        # 结果
        z1 = tf.add(tf.matmul(sum_embed, W1), b1)
        a1 = tf.nn.relu(z1)

        logits = tf.add(tf.matmul(a1, W2), b2)
        predictions = tf.nn.softmax(logits, name="predictions")
        predict_value = tf.cast(tf.argmax(predictions, 1), tf.int64)

    # loss
    with tf.name_scope("loss"):
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=logits))

    # optimizer
    with tf.name_scope("optimizer"):
        optimizer = tf.compat.v1.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

    # evaluation
    with tf.name_scope("evaluation"):
        correct_preds = tf.equal(tf.cast(tf.argmax(predictions, 1), tf.int64), targets)
        accuracy = tf.reduce_sum(tf.reduce_sum(tf.cast(correct_preds, tf.float32), axis=0))


sess = tf.compat.v1.Session()
init = tf.compat.v1.global_variables_initializer()
sess.run(init)
saver = tf.compat.v1.train.import_meta_graph('sent_predictor/dnn_checkpoints/dnn.meta')
saver.restore(sess, "sent_predictor/dnn_checkpoints/dnn")


def test_sentiment(predic_tokens, output_file=None):
    sentiments_count = {0: 0, 1: 0, 2: 0}

    for sentence in predic_tokens:
        input_tokens = np.array(sentence, dtype=np.int64).reshape(1, -1)
        pred_val = sess.run(predict_value, feed_dict={inputs: input_tokens})
        sentiments_count[pred_val[0]] += 1
        words = "".join([token_to_word[i] for i in sentence])
        # print("out is {} {}".format(words, pred_val[0]))
        if output_file:
            output_file.write('{} {}\n'.format(words, pred_val[0]))
        # print("Predic {} emotion is {}".format(words, sentiments[pred_val[0]]))
        # print("The sentimen of {} is {}".format(line, sentiments[pred_val[0]]))
    return sentiments_count[0], sentiments_count[1], sentiments_count[2]
