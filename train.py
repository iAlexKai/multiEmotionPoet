# 这段文字已经上传到87服务器上
import argparse
import time
from datetime import datetime
import numpy as np
import random
import json
import logging
import torch
import os, sys
from beeprint import pp

from configs import Config as Config
from data_apis.corpus import LoadPoem
from data_apis.data_utils import SWDADataLoader
from models.seq2seq import Seq2Seq
from models.poemwae import PoemWAE

from helper import to_tensor, timeSince  # 将numpy转为tensor

from experiments.metrics import Metrics
from sample import evaluate
from tensorboardX import SummaryWriter # install tensorboardX (pip install tensorboardX) before importing this package


parentPath = os.path.abspath("..")
sys.path.insert(0, parentPath)# add parent folder to path so as to import common modules

parser = argparse.ArgumentParser(description='headHider Pytorch')

# 大古诗数据集
parser.add_argument('--train_data_dir', type=str, default='./data/train_data.txt',
                    help='addr of data corpus for train and valid')

parser.add_argument('--test_data_dir', type=str, default='./data/test_data.txt',
                    help='addr of data for testing, i.e. test titles')

parser.add_argument('--max_vocab_size', type=int, default=10000, help='The size of the vocab, Cannot be None')
parser.add_argument('--expname', type=str, default='basic',
                    help='experiment name, for disinguishing different parameter settings')
parser.add_argument('--model', type=str, default='WAE', help='name of the model')
parser.add_argument('--visual', action='store_true', default=False, help='visualize training status in tensorboard')
parser.add_argument('--reload_from', type=int, default=-1, help='reload from a trained ephoch')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')

# Evaluation Arguments
parser.add_argument('--sample', action='store_true', help='sample when decoding for generation')
parser.add_argument('--log_every', type=int, default=50, help='interval to log training results')
parser.add_argument('--valid_every', type=int, default=50, help='interval to validation')
parser.add_argument('--eval_every', type=int, default=1, help='interval to evaluate on the validation set')
parser.add_argument('--test_every', type=int, default=1, help='interval to test on the titles')
parser.add_argument('--seed', type=int, default=1111, help='random seed')
parser.add_argument('--forward_only', default=False, action='store_true', help='only test, no training')

args = parser.parse_args()

# pretrain is True and test_align is False: 用五个小数据集训练从混合高斯分离出来的五个高斯分布，包含test
# pretrain is False and test_align is True: 测试大数据集训练出来的混合高斯中情感的align情况
# pretrain is False and test_align is False: 用大数据集训练5个高斯混合出来的模型

# if args.merge:
#     assert args.pretrain is False and args.test_align is False
# if args.divide:
#     assert args.pretrain is True and args.test_align is False

# if args.pretrain:
#     assert args.sentiment_path == '../final_data/poem_with_sentiment.txt'
#     assert args.test_align is False
#     assert args.dataset == 'SentimentPoems'

# if args.test_align:
#     assert args.dataset == 'TSPoems'
#     assert args.sentiment_path == '../final_data/poem_with_sentiment.txt'




# make output directory if it doesn't already exist

# Set the random seed manually for reproducibility.
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
use_cuda = torch.cuda.is_available()

if use_cuda:
    torch.cuda.set_device(args.gpu_id)  # set gpu device
    torch.cuda.manual_seed(args.seed)


# 从'./output/PoemWAE/sent_pretrain/SentimentPoems/{}/models/model_epo14.pckl' 提取先验分布参数组合成一个高斯混合分布
# 导入PoemWAE_GMP的先验分布区，作为初始化模型
def merge_to_mix_model():
    print("Load parameters from sub Gassian model to initialize the GMP")
    state_1 = torch.load(f='./output/PoemWAE/sent_pretrain/SentimentPoems/1/models/model_epo14.pckl')
    state_2 = torch.load(f='./output/PoemWAE/sent_pretrain/SentimentPoems/2/models/model_epo14.pckl')
    state_3 = torch.load(f='./output/PoemWAE/sent_pretrain/SentimentPoems/3/models/model_epo14.pckl')
    state_4 = torch.load(f='./output/PoemWAE/sent_pretrain/SentimentPoems/4/models/model_epo14.pckl')
    state_5 = torch.load(f='./output/PoemWAE/sent_pretrain/SentimentPoems/5/models/model_epo14.pckl')


def divide_to_sub_models():
    pass


def save_model(model, epoch, global_iter, batch_idx, sentiment=None):
    print("Saving model")
    # for item in model.state_dict():
    #     print(item)
    # import pdb
    # pdb.set_trace()
    if sentiment is None:
        torch.save(f='./output/{}/model_iter{}_epoch{}_batch{}.pckl'
                   .format(args.expname, global_iter, epoch, batch_idx),
                   obj=model)
    else:
        torch.save(f='./output/{}/model_iter{}_sent{}_epoch{}.pckl'
                   .format(args.model, args.expname, global_iter,
                           sentiment+1, epoch), obj=model)


# def save_sentiment_model(model, epoch, sentiment_id):
#     print("Saving sentiment model {}".format(sentiment_id))
#     for item in model.state_dict():
#         print(item)
#     torch.save(f='./output/{}/{}/{}/{}/models/model_epo{}.pckl'.format(args.model, args.expname,
#                                                     args.dataset, sentiment_id, epoch), obj=model)

def load_model(global_iter, epoch):
    print("Load model global iter{}, epoch{}".format(global_iter, epoch))
    model = torch.load(f='./output/{}/{}/{}/models/model_iter{}_epoch{}.pckl'.
                       format(args.model, args.expname, args.dataset, global_iter, epoch))
    model = model.cuda()
    return model


def load_model(model_path):
    model = torch.load(f='{}'.format(model_path))
    model = model.cuda()
    return model


def load_sentiment_model(iter, sentiment_index, epoch):
    print("Loading sentiment models iter {} sent {} epoch{}".format(iter, sentiment_index, epoch))
    model = torch.load(f='./output/{}/{}/{}/models/model_iter{}_sent{}_epoch{}.pckl'.
                       format(args.model, args.expname, args.dataset, iter, sentiment_index, epoch))
    model = model.cuda()
    return model


def process_pretrain_vec(pretrain_vec, vocab):
    pretrain_weight = []
    embed_dim = len(pretrain_vec['一'])
    for word in vocab:
        if word in pretrain_vec:
            pretrain_weight.append(np.array(pretrain_vec[word]))
        else:
            pretrain_weight.append(np.random.randn(embed_dim))
    return np.array(pretrain_weight)


def get_user_input(rev_vocab, title_size):

    def _is_Chinese(title):
        for ch in title:
            if '\u4e00' <= ch <= '\u9fff':
                return True
        return False

    while True:
        # title = str(input("请输入你要写的四字藏头诗（必须为四个字的中文）"))
        empty = True
        title = None
        while empty:
            try:
                with open('./content_from_local.txt', 'r') as input_file:
                    content = input_file.read()
                    if len(content) != 0:
                        empty = False
                        title = content[0:4]
            except:
                continue

        if title is None or title is "" or len(title) != 4 or not _is_Chinese(title):
            continue
        else:
            break

    title = [rev_vocab.get(item, rev_vocab["<unk>"]) for item in title]
    title_batch = [title + [0] * (title_size - len(title))]

    headers_batch = []
    for i in range(4):
        headers_batch.append([[title[i]]])

    return np.array(title_batch), headers_batch, title


def main():
    # config for training
    config = Config()
    print("Normal train config:")
    # pp(config)

    valid_config = Config()
    valid_config.dropout = 0
    valid_config.batch_size = 20

    # config for test
    test_config = Config()
    test_config.dropout = 0
    test_config.batch_size = 1

    # LOG #
    if not os.path.isdir('./output'):
        os.makedirs('./output')
    if not os.path.isdir('./output/{}'.format(args.expname)):
        os.makedirs('./output/{}'.format(args.expname))

    cur_time = str(datetime.now().strftime('%Y%m%d%H%M'))
    # save arguments
    json.dump(vars(args), open('./output/{}/{}_args.json'
                               .format(args.expname, cur_time), 'w'))

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG, format="%(message)s")
    fh = logging.FileHandler("./output/{}/logs_{}.txt"
                      .format(args.expname, cur_time))
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.info(vars(args))

    ###############################################################################
    # Load data
    ###############################################################################
    # sentiment data path:  ../ final_data / poem_with_sentiment.txt
    # 该path必须命令行显示输入LoadPoem，因为defaultNone
    # 处理pretrain数据和完整诗歌数据
    api = LoadPoem(args.train_data_dir, args.test_data_dir, args.max_vocab_size)

    # 交替训练，准备大数据集
    poem_corpus = api.get_poem_corpus()  # corpus for training and validation
    test_data = api.get_test_corpus()  # 测试数据
    # 三个list，每个list中的每一个元素都是 [topic, last_sentence, current_sentence]
    train_poem, valid_poem, test_poem = poem_corpus.get("train"), poem_corpus.get("valid"), test_data.get("test")

    train_loader = SWDADataLoader("Train", train_poem, config)
    valid_loader = SWDADataLoader("Valid", valid_poem, config)
    test_loader = SWDADataLoader("Test", test_poem, config)

    print("Finish Poem data loading, not pretraining or alignment test")

    if not args.forward_only:
        ###############################################################################
        # Define the models and word2vec weight
        ###############################################################################
        # 处理用四库全书训练的word2vec
        # if args.model != "Seq2Seq"

        # logger.info("Start loading siku word2vec")
        # pretrain_weight = None
        # if os.path.exists(args.word2vec_path):
        #     pretrain_vec = {}
        #     word2vec = open(args.word2vec_path)
        #     pretrain_data = word2vec.read().split('\n')[1:]
        #     for data in pretrain_data:
        #         data = data.split(' ')
        #         pretrain_vec[data[0]] = [float(item) for item in data[1:-1]]
        #     # nparray (vocab_len, emb_dim)
        #     pretrain_weight = process_pretrain_vec(pretrain_vec, api.vocab)
        #     logger.info("Successfully loaded siku word2vec")

        # import pdb
        # pdb.set_trace()

        # 无论是否pretrain，都使用高斯混合模型
        # pretrain时，用特定数据训练特定的高斯分布
        # 不用pretrain时，用大数据训练高斯混合分布

        if args.model == "Seq2Seq":
            model = Seq2Seq(config=config, api=api)
        else:
            model = PoemWAE(config=config, api=api)

        if use_cuda:
            model = model.cuda()
        # if corpus.word2vec is not None and args.reload_from<0:
        #     print("Loaded word2vec")
        #     model.embedder.weight.data.copy_(torch.from_numpy(corpus.word2vec))
        #     model.embedder.weight.data[0].fill_(0)

        ###############################################################################
        # Start training
        ###############################################################################
        # model依然是PoemWAE_GMP保持不变，只不过，用这部分数据强制训练其中一个高斯先验分布
        # pretrain = True

        tb_writer = SummaryWriter(
            "./output/{}/{}/{}/logs/".format(args.model, args.expname, args.dataset)\
            + datetime.now().strftime('%Y%m%d%H%M')) if args.visual else None

        global_iter = 1
        cur_best_score = {'min_valid_loss': 100, 'min_global_itr': 0, 'min_epoch': 0, 'min_itr': 0}

        train_loader.epoch_init(config.batch_size, shuffle=True)

        # model = load_model(3, 3)
        batch_idx = 0
        while global_iter < 100:
            batch_idx = 0
            while True:  # loop through all batches in training data
                # train一个batch
                model, finish_train, loss_records = \
                    train_process(model=model, train_loader=train_loader, config=config, sentiment_data=False)

                batch_idx += 1
                if finish_train:
                    test_process(model=model, test_loader=test_loader, test_config=test_config, logger=logger)
                    evaluate_process(model=model, valid_loader=valid_loader, global_iter=global_iter,
                                     epoch=global_iter, logger=logger, tb_writer=tb_writer, api=api)
                    # save model after each epoch
                    save_model(model=model, epoch=global_iter, global_iter=global_iter, batch_idx=batch_idx)
                    logger.info('Finish epoch %d, current min valid loss: %.4f \
                     correspond global_itr: %d  epoch: %d  itr: %d \n\n' % (global_iter,
                                   cur_best_score['min_valid_loss'], cur_best_score['min_global_itr'],
                                   cur_best_score['min_epoch'], cur_best_score['min_itr']))
                    # 初始化下一个unlabeled data epoch的训练
                    # unlabeled_epoch += 1
                    train_loader.epoch_init(config.batch_size, shuffle=True)
                    break
                # elif batch_idx >= start_batch + config.n_batch_every_iter:
                #     print("Finish unlabel epoch %d batch %d to %d" %
                #           (unlabeled_epoch, start_batch, start_batch + config.n_batch_every_iter))
                #     start_batch += config.n_batch_every_iter
                #     break

                # 写一下log
                if batch_idx % (train_loader.num_batch // 50) == 0:
                    log = 'Global iter %d: step: %d/%d: ' \
                          % (global_iter, batch_idx, train_loader.num_batch)
                    for loss_name, loss_value in loss_records:
                        log = log + loss_name + ':%.4f ' % loss_value
                        if args.visual:
                            tb_writer.add_scalar(loss_name, loss_value, global_iter)
                    logger.info(log)

                # valid
                if batch_idx % (train_loader.num_batch // 10) == 0:
                    valid_process(model=model, valid_loader=valid_loader, valid_config=valid_config,
                                  global_iter=global_iter, unlabeled_epoch=global_iter,  # 如果sample_rate_unlabeled不是1，这里要在最后加一个1
                                  batch_idx=batch_idx, tb_writer=tb_writer, logger=logger,
                                  cur_best_score=cur_best_score)
                    test_process(model=model, test_loader=test_loader, test_config=test_config, logger=logger)
                    save_model(model=model, epoch=global_iter, global_iter=global_iter, batch_idx=batch_idx)
                # if batch_idx % (train_loader.num_batch // 3) == 0:
                #     test_process(model=model, test_loader=test_loader, test_config=test_config, logger=logger)

            global_iter += 1

    # forward_only 测试
    else:
        # test_global_list = [4, 4, 2]
        # test_epoch_list = [21, 19, 8]
        test_global_list = [8]
        test_epoch_list = [20]
        for i in range(1):
            # import pdb
            # pdb.set_trace()
            model = load_model('./output/basic/header_model.pckl')
            model.vocab = api.vocab
            model.rev_vocab = api.rev_vocab
            test_loader.epoch_init(test_config.batch_size, shuffle=False)

            last_title = None
            while True:
                model.eval()  # eval()主要影响BatchNorm, dropout等操作
                batch = get_user_input(api.rev_vocab, config.title_size)

                # batch = test_loader.next_batch_test()  # test data使用专门的batch
                # import pdb
                # pdb.set_trace()
                if batch is None:
                    break

                title_list, headers, title = batch  # batch size是1，一个batch写一首诗

                if title == last_title:
                    continue
                last_title = title

                title_tensor = to_tensor(title_list)

                # test函数将当前batch对应的这首诗decode出来，记住每次decode的输入context是上一次的结果
                output_poem = model.test(title_tensor=title_tensor, title_words=title_list, headers=headers)
                with open('./content_from_remote.txt', 'w') as file:
                    file.write(output_poem)
                print(output_poem)
                print('\n')
            print("Done testing")


def train_process(model, train_loader, config, sentiment_data=False, mask_type=None):
    model.train()
    loss_records = []
    sentiment_mask = None
    if sentiment_data:

        batch = train_loader.next_sentiment_batch()
        finish_train = False
        if batch is None:  # end of epoch
            finish_train = True
            return model, finish_train, None
        title, context, target, target_lens, sentiment_mask = batch
        title, context, target, target_lens, sentiment_mask = \
            to_tensor(title), to_tensor(context), to_tensor(target), to_tensor(target_lens), to_tensor(sentiment_mask)
    else:
        batch = train_loader.next_batch()
        finish_train = False
        if batch is None:  # end of epoch
            finish_train = True
            return model, finish_train, None
        title, context, target, target_lens = batch
        title, context, target, target_lens = \
            to_tensor(title), to_tensor(context), to_tensor(target), to_tensor(target_lens)

    # import pdb
    # pdb.set_trace()
    loss_AE = model.train_AE(title, context, target, target_lens)  # 输入topic，last句，当前句，当前句长度
    loss_records.extend(loss_AE)

    loss_G = model.train_G(title, context, target, target_lens, sentiment_mask=sentiment_mask, mask_type=mask_type)
    loss_records.extend(loss_G)

    # 训练 Discriminator
    for i in range(config.n_iters_d):  # train discriminator/critic
        loss_D = model.train_D(title, context, target, target_lens)
        if i == 0:
            loss_records.extend(loss_D)
        if i == config.n_iters_d - 1:
            break
        batch = train_loader.sample_one_batch(sentiment=sentiment_data)
        if batch is None:  # end of epoch
            break
        title, context, target, target_lens = batch
        title, context, target, target_lens = \
            to_tensor(title), to_tensor(context), to_tensor(target), to_tensor(target_lens)

    return model, finish_train, loss_records


def evaluate_process(model, valid_loader, global_iter, epoch, logger, tb_writer, api):
    model.eval()
    valid_loader.epoch_init(1, shuffle=False)  # batch_size是1，重复10次，计算BLEU

    f_eval = open(
        "./output/{}/eval_global_{}_epoch{}.txt".format(args.expname, global_iter, epoch), "w")
    repeat = 10

    # 测试当前model
    # Define the metrics
    metrics = Metrics(model.embedder)
    recall_bleu, prec_bleu, bow_extrema, bow_avg, bow_greedy, intra_dist1, intra_dist2, avg_len, inter_dist1, inter_dist2 \
        = evaluate(model, metrics, valid_loader, api.vocab, api.rev_vocab, f_eval, repeat)

    logger.info("Avg recall BLEU %f, avg precision BLEU %f, bow_extrema %f, bow_avg %f, bow_greedy %f, intra_dist1 %f,"
                " intra_dist2 %f, avg_len %f, \ninter_dist1 %f, inter_dist2 %f (only 1 ref, not final results)" \
                % (recall_bleu, prec_bleu, bow_extrema, bow_avg, bow_greedy, intra_dist1, intra_dist2, avg_len,
                   inter_dist1, inter_dist2))

    if args.visual:
        tb_writer.add_scalar('recall_bleu', recall_bleu, epoch)
        tb_writer.add_scalar('prec_bleu', prec_bleu, epoch)
        tb_writer.add_scalar('bow_extrema', bow_extrema, epoch)
        tb_writer.add_scalar('bow_avg', bow_avg, epoch)
        tb_writer.add_scalar('bow_greedy', bow_greedy, epoch)
        tb_writer.add_scalar('intra_dist1', intra_dist1, epoch)
        tb_writer.add_scalar('intra_dist2', intra_dist2, epoch)
        tb_writer.add_scalar('inter_dist1', inter_dist1, epoch)
        tb_writer.add_scalar('inter_dist2', inter_dist2, epoch)


def test_process(model, test_loader, test_config, logger):
    # 训练完一个epoch，用测试集的标题生成一次诗

    # mask_types = ['negative', 'positive', 'neutral']
    model.eval()
    output_poems = ""
    test_loader.epoch_init(test_config.batch_size, shuffle=False)
    while True:
        model.eval()  # eval()主要影响BatchNorm, dropout等操作
        batch = test_loader.next_batch_test()  # test data使用专门的batch
        if batch is None:
            break
        title_list, headers = batch  # batch size是1，一个batch写一首诗
        title_tensor = to_tensor(title_list)

        # test函数将当前batch对应的这首诗decode出来，记住每次decode的输入context是上一次的结果
        # output_poem = 'Global iter: {}\n'.format(global_iter)
        output_poem = model.test(title_tensor=title_tensor, title_words=title_list, headers=headers)
        output_poems += output_poem
    logger.info(output_poems)

    print("Done testing")


def valid_process(model, valid_loader, valid_config, global_iter, unlabeled_epoch, batch_idx,
                  tb_writer, logger, cur_best_score):
    valid_loader.epoch_init(valid_config.batch_size, shuffle=False)
    model.eval()
    loss_records = {}

    while True:
        batch = valid_loader.next_batch()
        if batch is None:  # end of epoch
            break

        title, context, target, target_lens = batch
        title, context, target, target_lens = \
            to_tensor(title), to_tensor(context), to_tensor(target), to_tensor(target_lens)
        valid_loss = model.valid(title, context, target, target_lens)
        for loss_name, loss_value in valid_loss:
            v = loss_records.get(loss_name, [])
            if loss_name == 'min_valid_loss' and loss_value < cur_best_score['min_valid_loss']:
                cur_best_score['min_valid_loss'] = loss_value
                cur_best_score['min_global_itr'] = global_iter
                cur_best_score['min_epoch'] = unlabeled_epoch
                cur_best_score['min_itr'] = batch_idx

            v.append(loss_value)
            loss_records[loss_name] = v

    log = 'Global iter {} Validation:'.format(global_iter)
    for loss_name, loss_values in loss_records.items():
        # import pdb
        # pdb.set_trace()
        log = log + loss_name + ':%.4f  ' % (np.mean(loss_values))
        if args.visual:
            tb_writer.add_scalar(loss_name, np.mean(loss_values), global_iter)

    logger.info(log)


def valid_process_sentiment(model, valid_poem_loader, valid_config, global_iter, num,
                  tb_writer, logger, cur_best_score_labeled):
    valid_poem_loader.epoch_init(valid_config.batch_size, shuffle=False)
    model.eval()
    loss_records = {}
    while True:
        batch = valid_poem_loader.next_sentiment_batch()
        if batch is None:  # end of epoch
            break

        title, context, target, target_lens, sentiment_mask = batch
        title, context, target, target_lens, sentiment_mask = \
            to_tensor(title), to_tensor(context), to_tensor(target), to_tensor(target_lens), to_tensor(sentiment_mask)
        valid_loss = model.valid(title, context, target, target_lens, sentiment_mask)
        for loss_name, loss_value in valid_loss:
            v = loss_records.get(loss_name, [])
            v.append(loss_value)
            loss_records[loss_name] = v

    log = 'Valid: Global iter {} Validation\n'.format(global_iter)
    for loss_name, loss_values in loss_records.items():
        # import pdb
        # pdb.set_trace()
        if loss_name == 'valid_loss_AE' and np.mean(loss_values) < cur_best_score_labeled['min_valid_loss_label']:
            log += "\nFOUND a new best valid loss in global %d, num %d\n" % (global_iter, num)
            cur_best_score_labeled['min_valid_loss_label'] = np.mean(loss_values)
            cur_best_score_labeled['min_global_itr_label'] = global_iter
            cur_best_score_labeled['min_num_label'] = num

        log = log + loss_name + ':%.4f  ' % (np.mean(loss_values))
        if args.visual:
            tb_writer.add_scalar(loss_name, np.mean(loss_values), global_iter)

    logger.info(log)


# def update_poem_dataloader(sentiment_poem_corpus, config, global_itr):
#     poem_1 = sentiment_poem_corpus.get("sent_1")  # 831  / 1    831
#     poem_2 = sentiment_poem_corpus.get("sent_2")  # 4045 / 1    4045  4876
#     poem_3 = sentiment_poem_corpus.get("sent_3")  # 6989 / 1.5  4660  4660
#     poem_4 = sentiment_poem_corpus.get("sent_4")  # 4557 / 1.2  3800  4580
#     poem_5 = sentiment_poem_corpus.get("sent_5")  # 934  /1.2   780
#
#     # 831 4045 4876 4199 776
#     # n_neutral = int(4876 * (1 - global_itr / 5))
#     # n_neutral = max(0, n_neutral)
#     poems = random.sample(poem_1, 700)\
#             + random.sample(poem_2, 4000)\
#             + random.sample(poem_3, 4700)\
#             + random.sample(poem_4, 4000)\
#             + random.sample(poem_5, 700)
#     valid_poems_len = len(poems) // 10
#     train_poems = poems[:-valid_poems_len]
#     valid_poems = poems[-valid_poems_len:]
#     random.shuffle(train_poems)
#     random.shuffle(valid_poems)
#     train_poem_loader = SWDADataLoader("Sentiment_Poem_Train", train_poems, config)
#     valid_poem_loader = SWDADataLoader("Sentiment_Poem_Valid", valid_poems, config)
#     return train_poem_loader, valid_poem_loader


if __name__ == "__main__":
    main()


