import argparse
from beeprint import pp
from datetime import datetime
import numpy as np
import random
import json
import logging
import torch
import os, sys

from configs import Config as Config
from data_apis.corpus import LoadPoem
from data_apis.data_utils import SWDADataLoader
from models.seq2seq import Seq2Seq
from models.cvae import CVAE
from models.cvae_gmp import CVAE_GMP
from helper import test_sentiment, write_predict_result_to_file, to_tensor# 将numpy转为tensor

from experiments.metrics import Metrics
from sample import evaluate
from tensorboardX import SummaryWriter # install tensorboardX (pip install tensorboardX) before importing this package


parentPath = os.path.abspath("..")
sys.path.insert(0, parentPath)# add parent folder to path so as to import common modules

parser = argparse.ArgumentParser(description='headHider Pytorch')

# # 大古诗数据集
# parser.add_argument('--train_data_dir', type=str, default='./data/train_data.txt',
#                     help='addr of data corpus for train and valid')

# 大古诗及预测情感后的数据集
parser.add_argument('--train_data_dir', type=str, default='./data/train_data_with_sent.txt',
                    help='train data with the predicted sentiments')

parser.add_argument('--test_data_dir', type=str, default='./data/test_data.txt',
                    help='addr of data for testing, i.e. test titles')
parser.add_argument('--expname', type=str, default='sentInput',
                    help='experiment name, for disinguishing different parameter settings')
# parser.add_argument('--model', type=str, default='mCVAE', help='name of the model')
parser.add_argument('--model', type=str, default='CVAE', help='name of the model')
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


def save_model(model, epoch, log_start_time, global_t, sentiment=None):
    print("Saving model")
    # for item in model.state_dict():
    #     print(item)
    # import pdb
    # pdb.set_trace()
    if sentiment is None:
        torch.save(f='./output/{}/{}/model_global_t_{}_epoch{}.pckl'
                   .format(args.expname, log_start_time, global_t, epoch),
                   obj=model)
    else:
        torch.save(f='./output/{}/{}/model_iter{}_sent{}_epoch{}.pckl'
                   .format(args.expname, log_start_time, global_t,
                           sentiment+1, epoch), obj=model)


# def save_sentiment_model(model, epoch, sentiment_id):
#     print("Saving sentiment model {}".format(sentiment_id))
#     for item in model.state_dict():
#         print(item)
#     torch.save(f='./output/{}/{}/{}/{}/models/model_epo{}.pckl'.format(args.model, args.expname,
#                                                     args.dataset, sentiment_id, epoch), obj=model)

def load_model(model_name):
    print("Load model {}".format(model_name))
    model = torch.load(f='{}'.format(model_name))
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
    pp(config)

    valid_config = Config()
    valid_config.dropout = 0
    valid_config.batch_size = 20

    # config for test
    test_config = Config()
    test_config.dropout = 0
    test_config.batch_size = 1

    with_sentiment = config.with_sentiment

    ###############################################################################
    # Load data
    ###############################################################################
    # sentiment data path:  ../ final_data / poem_with_sentiment.txt
    # 该path必须命令行显示输入LoadPoem，因为defaultNonehjk
    # 处理pretrain数据和完整诗歌数据

    # api = LoadPoem(args.train_data_dir, args.test_data_dir, args.max_vocab_size)
    api = LoadPoem(corpus_path=args.train_data_dir, test_path=args.test_data_dir, max_vocab_cnt=config.max_vocab_cnt,
                   with_sentiment=with_sentiment)

    # 交替训练，准备大数据集
    poem_corpus = api.get_tokenized_poem_corpus(type=1+int(with_sentiment))  # corpus for training and validation
    test_data = api.get_tokenized_test_corpus()  # 测试数据
    # 三个list，每个list中的每一个元素都是 [topic, last_sentence, current_sentence]
    train_poem, valid_poem, test_poem = poem_corpus["train"], poem_corpus["valid"], test_data["test"]

    train_loader = SWDADataLoader("Train", train_poem, config)
    valid_loader = SWDADataLoader("Valid", valid_poem, config)
    test_loader = SWDADataLoader("Test", test_poem, config)

    print("Finish Poem data loading, not pretraining or alignment test")

    if not args.forward_only:
        # LOG #
        log_start_time = str(datetime.now().strftime('%Y%m%d%H%M'))
        if not os.path.isdir('./output'):
            os.makedirs('./output')
        if not os.path.isdir('./output/{}'.format(args.expname)):
            os.makedirs('./output/{}'.format(args.expname))
        if not os.path.isdir('./output/{}/{}'.format(args.expname, log_start_time)):
            os.makedirs('./output/{}/{}'.format(args.expname, log_start_time))

        # save arguments
        json.dump(vars(args), open('./output/{}/{}/args.json'
                                   .format(args.expname, log_start_time), 'w'))

        logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.DEBUG, format="%(message)s")
        fh = logging.FileHandler("./output/{}/{}/logs.txt"
                                 .format(args.expname, log_start_time))
        # add the handlers to the logger
        logger.addHandler(fh)
        logger.info(vars(args))

        tb_writer = SummaryWriter(
            "./output/{}/{}/tb_logs".format(args.expname, log_start_time)) if args.visual else None

        if config.reload_model:
            model = load_model(config.model_name)
        else:
            if args.model == "mCVAE":
                model = CVAE_GMP(config=config, api=api)
            elif args.model == 'CVAE':
                model = CVAE(config=config, api=api)
            else:
                model = Seq2Seq(config=config, api=api)
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

        cur_best_score = {'min_valid_loss': 100, 'min_global_itr': 0, 'min_epoch': 0, 'min_itr': 0}

        train_loader.epoch_init(config.batch_size, shuffle=True)

        # model = load_model(3, 3)
        epoch_id = 0
        global_t = 0
        while epoch_id < config.epochs:

            while True:  # loop through all batches in training data
                # train一个batch
                model, finish_train, loss_records, global_t = \
                    train_process(global_t=global_t, model=model, train_loader=train_loader, config=config, sentiment_data=with_sentiment)
                if finish_train:
                    test_process(model=model, test_loader=test_loader, test_config=test_config, logger=logger)
                    # evaluate_process(model=model, valid_loader=valid_loader, log_start_time=log_start_time, global_t=global_t, epoch=epoch_id, logger=logger, tb_writer=tb_writer, api=api)
                    # save model after each epoch
                    save_model(model=model, epoch=epoch_id, global_t=global_t, log_start_time=log_start_time)
                    logger.info('Finish epoch %d, current min valid loss: %.4f \
                     correspond epoch: %d  itr: %d \n\n' % (cur_best_score['min_valid_loss'], cur_best_score['min_global_itr'],
                                   cur_best_score['min_epoch'], cur_best_score['min_itr']))
                    # 初始化下一个unlabeled data epoch的训练
                    # unlabeled_epoch += 1
                    epoch_id += 1
                    train_loader.epoch_init(config.batch_size, shuffle=True)
                    break
                # elif batch_idx >= start_batch + config.n_batch_every_iter:
                #     print("Finish unlabel epoch %d batch %d to %d" %
                #           (unlabeled_epoch, start_batch, start_batch + config.n_batch_every_iter))
                #     start_batch += config.n_batch_every_iter
                #     break

                # 写一下log
                if global_t % config.log_every == 0:
                    log = 'Epoch id %d: step: %d/%d: ' \
                          % (epoch_id, global_t % train_loader.num_batch, train_loader.num_batch)
                    for loss_name, loss_value in loss_records:
                        if loss_name == 'avg_lead_loss':
                            continue
                        log = log + loss_name + ':%.4f ' % loss_value
                        if args.visual:
                            tb_writer.add_scalar(loss_name, loss_value, global_t)
                    logger.info(log)

                # valid
                if global_t % config.valid_every == 0:
                    # test_process(model=model, test_loader=test_loader, test_config=test_config, logger=logger)
                    valid_process(global_t=global_t, model=model, valid_loader=valid_loader, valid_config=valid_config,
                                  unlabeled_epoch=epoch_id,  # 如果sample_rate_unlabeled不是1，这里要在最后加一个1
                                  tb_writer=tb_writer, logger=logger,
                                  cur_best_score=cur_best_score)
                # if batch_idx % (train_loader.num_batch // 3) == 0:
                #     test_process(model=model, test_loader=test_loader, test_config=test_config, logger=logger)
                if global_t % config.test_every == 0:
                    test_process(model=model, test_loader=test_loader, test_config=test_config, logger=logger)

    # forward_only 测试
    else:
        expname = 'sentInput'
        time = '202101191105'

        model = load_model('./output/{}/{}/model_global_t_13596_epoch3.pckl'.format(expname, time))
        test_loader.epoch_init(test_config.batch_size, shuffle=False)
        if not os.path.exists('./output/{}/{}/test/'.format(expname, time)):
            os.mkdir('./output/{}/{}/test/'.format(expname, time))
        output_file = [open('./output/{}/{}/test/output_0.txt'
                           .format(expname, time),
                           'w'),
                       open('./output/{}/{}/test/output_1.txt'
                           .format(expname, time),
                           'w'),
                       open('./output/{}/{}/test/output_2.txt'
                           .format(expname, time),
                           'w')]

        poem_count = 0
        predict_results = {0: [], 1: [], 2: []}
        titles = {0: [], 1: [], 2: []}
        sentiment_result = {0: [], 1: [], 2: []}
        # Get all poem predictions
        while True:
            model.eval()
            batch = test_loader.next_batch_test()  # test data使用专门的batch
            poem_count += 1
            if poem_count % 10 == 0:
                print("Predicted {} poems".format(poem_count))
            if batch is None:
                break
            title_list = batch  # batch size是1，一个batch写一首诗
            title_tensor = to_tensor(title_list)
            # test函数将当前batch对应的这首诗decode出来，记住每次decode的输入context是上一次的结果
            for i in range(3):
                sentiment_label = np.zeros(1, dtype=np.int64)
                sentiment_label[0] = int(i)
                sentiment_label = to_tensor(sentiment_label)
                output_poem, output_tokens = model.test(title_tensor, title_list, sentiment_label=sentiment_label)

                titles[i].append(output_poem.strip().split('\n')[0])
                predict_results[i] += (np.array(output_tokens)[:, :7].tolist())

        # Predict sentiment use the sort net
        from collections import defaultdict
        neg = defaultdict(int)
        neu = defaultdict(int)
        pos = defaultdict(int)
        total = defaultdict(int)
        for i in range(3):
            _, neg[i], neu[i], pos[i] = test_sentiment(predict_results[i])
            total[i] = neg[i] + neu[i] + pos[i]

        for i in range(3):
            print("%d%%\t%d%%\t%d%%" % (neg * 100 / total, neu * 100 / total, pos * 100 / total))

        for i in range(3):
            write_predict_result_to_file(titles[i], predict_results[i], sentiment_result[i], output_file[i])
            output_file[i].close()

        print("Done testing")


def train_process(global_t, model, train_loader, config, sentiment_data=False):
    model.train()
    loss_records = []
    sentiment_label = None
    if sentiment_data:
        batch = train_loader.next_sentiment_batch()
        finish_train = False
        if batch is None:  # end of epoch
            finish_train = True
            return model, finish_train, None, global_t
        title, context, target, target_lens, sentiment_label = batch
        title, context, target, target_lens, sentiment_label = \
            to_tensor(title), to_tensor(context), to_tensor(target), to_tensor(target_lens), to_tensor(sentiment_label)
    else:
        batch = train_loader.next_batch()
        finish_train = False
        if batch is None:  # end of epoch
            finish_train = True
            return model, finish_train, None, global_t
        title, context, target, target_lens = batch
        title, context, target, target_lens = \
            to_tensor(title), to_tensor(context), to_tensor(target), to_tensor(target_lens)

    # global_t, title, context, target, target_lens,
    loss_AE, global_t = model.train_AE(global_t, title, context, target, target_lens, sentiment_label)  # 输入topic，last句，当前句，当前句长度
    loss_records.extend(loss_AE)

    return model, finish_train, loss_records, global_t


def valid_process(global_t, model, valid_loader, valid_config, unlabeled_epoch,
                  tb_writer, logger, cur_best_score):
    valid_loader.epoch_init(valid_config.batch_size, shuffle=False)
    model.eval()
    loss_records = {}
    while True:
        batch = valid_loader.next_sentiment_batch()
        if batch is None:  # end of epoch
            break
        title, context, target, target_lens, sentiment_label = batch
        title, context, target, target_lens, sentiment_label = \
            to_tensor(title), to_tensor(context), to_tensor(target), to_tensor(target_lens), to_tensor(
                sentiment_label)
        valid_loss = model.valid_AE(global_t, title, context, target, target_lens, sentiment_label)
        for loss_name, loss_value in valid_loss:
            v = loss_records.get(loss_name, [])
            if loss_name == 'min_valid_loss' and loss_value < cur_best_score['min_valid_loss']:
                cur_best_score['min_valid_loss'] = loss_value
                cur_best_score['min_epoch'] = unlabeled_epoch
                cur_best_score['min_step'] = global_t

            v.append(loss_value)
            loss_records[loss_name] = v

    log = ""
    for loss_name, loss_values in loss_records.items():
        log = log + loss_name + ':%.4f  ' % (np.mean(loss_values))
        if args.visual:
            tb_writer.add_scalar(loss_name, np.mean(loss_values), global_t)

    logger.info(log)


def evaluate_process(model, valid_loader, log_start_time, global_t, epoch, logger, tb_writer, api):
    model.eval()
    valid_loader.epoch_init(1, shuffle=False)  # batch_size是1，重复10次，计算BLEU

    f_eval = open(
        "./output/{}/{}/eval_global_{}_epoch{}.txt".format(args.expname, log_start_time, global_t, epoch), "w")
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

    test_loader.epoch_init(test_config.batch_size, shuffle=True)

    poem_count = 0
    predict_results = {0: [], 1: [], 2: []}
    while True:
        model.eval()
        batch = test_loader.next_batch_test()  # test data使用专门的batch
        if batch is None:
            break
        batch_size = batch.shape[0]
        poem_count += 1
        if poem_count % 10 == 0:
            print("Predicted {} poems".format(poem_count))
        if batch is None:
            break
        title_list = batch  # batch size是1，一个batch写一首诗
        title_tensor = to_tensor(title_list)
        # test函数将当前batch对应的这首诗decode出来，记住每次decode的输入context是上一次的结果

        for i in range(3):
            # import pdb
            # pdb.set_trace()
            sentiment_label = np.zeros(batch_size, dtype=np.int64)
            for id in range(batch_size):
                sentiment_label[id] = int(i)
            sentiment_label = to_tensor(sentiment_label)
            output_poem, output_tokens = model.test(title_tensor, title_list, sentiment_label=sentiment_label)

            if poem_count % 80 == 0:
                logger.info("Sentiment {} Poem {}\n".format(i, output_poem))

            predict_results[i] += (np.array(output_tokens)[:, :7].tolist())

    # Predict sentiment use the sort net
    from collections import defaultdict
    neg = defaultdict(int)
    neu = defaultdict(int)
    pos = defaultdict(int)
    total = defaultdict(int)
    for i in range(3):
        _, neg[i], neu[i], pos[i] = test_sentiment(predict_results[i])
        total[i] = neg[i] + neu[i] + pos[i]

    for i in range(3):
        logger.info("%d%%\t%d%%\t%d%%" %
              (neg[i] * 100 / total[i], neu[i] * 100 / total[i], pos[i] * 100 / total[i]))
    print("Done testing")


if __name__ == "__main__":
    main()


