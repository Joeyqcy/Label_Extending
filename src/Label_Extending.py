import logging
logger = logging.getLogger(__name__)
import os
import time
import re
import json
import datetime
import pandas as pd
import text_clean
from multiprocessing import Pool, Manager
from itertools import chain
from pyltp import Segmentor, Postagger, Parser, NamedEntityRecognizer, SentenceSplitter
import maddl

'''
路径的设置
'''
DATE = datetime.datetime.now().strftime('%m_%d')  # 测试日期
MODELDIR = '/home/share/ltp_data_v3.4.0'  # ltp模型路径
SEGMENTOR_DICT_PATH = '../datasets/newwords_dict.txt'  # 分词和词性标注词典
POSTAGGER_DICT_PATH = '../datasets/merge_pos_dict.txt'
LABELED_DICT = '../datasets/labeled_dict.txt'  # 已标签词典
DATABASE_NAME = 'kol_recommend'  # 数据库定位
TABLE_NAME = 'baidu_zhidao'
INPUT_CSV_PATH = '../datasets/%s_%s.csv' % (DATABASE_NAME, TABLE_NAME)  # 语料路径
OUTPUT_CSV_PATH = '../output/%s_Output.csv' % DATE  # 输出路径
NEWWORDS_CSV_PATH = '../output/%s_NewWords.csv' % DATE  # 新词表输出路径

GRAPH_FILE = '%s_graph.txt' % DATE
SEED_FILE = '%s_seed.txt' % DATE
POTENTIAL_FILE = 'Potential_list.json'
OUTPUT_FILE = '%s_output.txt' % DATE

'''
`POTENTIAL_LIST` 存放于文档中发现的潜在实体词
`LABELED_LIST` 存放已知的有标签实体词
`FILTER_LIST` 存放用于筛选图结构的词，为上述两个词集的并集
'''

POTENTIAL_LIST, LABELED_LIST, FILTER_LIST = [], [], []
TARGET_DICT = {}
FILTER_LIST_FILE = 'Filter_list.json'
CONTENT = 'content'  # 处理的列名
WORKERS = 16
_DOC_LENGTH_FFILTER = 8



'''
ltp模型加载
'''
segmentor = Segmentor()
segmentor.load_with_lexicon(os.path.join(MODELDIR, "cws.model"), SEGMENTOR_DICT_PATH)

postagger = Postagger()
postagger.load_with_lexicon(os.path.join(MODELDIR, "pos.model"), POSTAGGER_DICT_PATH)

parser = Parser()
parser.load(os.path.join(MODELDIR, "parser.model"))

recognizer = NamedEntityRecognizer()
recognizer.load(os.path.join(MODELDIR, "ner.model"))

flatten = lambda l: list(chain.from_iterable(l))



def _get_labeled(labeled_path=LABELED_DICT):
    global LABELED_LIST
    with open(labeled_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.split()
            LABELED_LIST.append(line[0])
    print('Len of LABELED_LIST %d' % len(LABELED_LIST))


def split2sent(text):
    sents = SentenceSplitter.split(text)
    return list(sents)


def conts2sents(contents, workers):
    with Pool(processes=workers) as _executor:
        SENTENCES = _executor.map(split2sent, contents)
    return flatten(SENTENCES)


def _segment(sent):
    return list(segmentor.segment(sent))


def _postag(words):
    return list(postagger.postag(words))


def _netag(words_postags):
    return list(recognizer.recognize(words_postags[0], words_postags[1]))


def _parse(words_postags):
    arcs = parser.parse(words_postags[0], words_postags[1])
    return [(arc.head, arc.relation) for arc in arcs]


def get_words(sents, workers):
    t1 = time.time()
    with Pool(processes=workers) as _executor:
        WORDS = _executor.map(_segment, sents)
    segmentor.release()
    print('Segment Finished, Time cost %f s' % (time.time() - t1))
    return WORDS


def get_postags(WORDS, workers):
    t1 = time.time()
    with Pool(processes=workers) as _executor:
        POSTAGS = _executor.map(_postag, WORDS)
    postagger.release()
    print('Postag Finished, Time cost %f s' % (time.time() - t1))
    return POSTAGS


def get_netags(WORDS, POSTAGS, workers):
    t1 = time.time()
    with Pool(processes=workers) as _executor:
        NETAGS = _executor.map(_netag, zip(WORDS, POSTAGS))
    recognizer.release()
    print('Netag Finished, Time cost %f s' % (time.time() - t1))
    return NETAGS


def get_arcs(WORDS,POSTAGS, workers):
    t1 = time.time()
    with Pool(processes=workers) as _executor:
        ARCS = _executor.map(_parse, zip(WORDS, POSTAGS))
    parser.release()
    print('Parse Finished, Time cost %f s' % (time.time() - t1))
    return ARCS


def _connect_entity(words, netags, index):
    '''
    连接散开的实体词
    '''
    if index == len(netags):
        return ''
    if netags[index] in ['B-Ni', 'B-Nh', 'I-Ni', 'I-Nh']:
        return words[index] + _connect_entity(words, netags, index + 1)
    elif netags[index] in ['E-Ni', 'E-Nh']:
        return words[index]


def _findNE_fromSent(words_info):
    '''
    `words_info`  (words,postags,netags)
    '''
    _potential_list = []
    words, postags, netags = words_info
    isEntity = lambda x: x in ['S-Ni', 'S-Nh']
    isB_Entity = lambda x: x in ['B-Ni', 'B-Nh']
    for index, word in enumerate(words):
        if isEntity(netags[index]) and len(word) > 1:  # 这里暂时先不考虑句法分析中的潜在实体词提取
            _potential_list.append(word)
        if isB_Entity(netags[index]):  # 将分散的实体整体连接起来
            _potential_list.append(_connect_entity(words, netags, index))
    _potential_list = list(set(_potential_list))
    return _potential_list


def _findNE_fromALL(WORDS, POSTAGS, NETAGS, workers):
    global POTENTIAL_LIST
    with Pool(processes=workers) as _executor:
        _potentials = _executor.map(_findNE_fromSent, zip(WORDS, POSTAGS, NETAGS))
    return _potentials


def build_parse_child_dict(words, postags, arcs):
    """
    为句子中的每个词语维护一个保存句法依存儿子节点的字典
    """
    child_dict_list = []
    for index in range(len(words)):
        child_dict = {}
        for arc_index in range(len(arcs)):
            if arcs[arc_index][0] == index + 1: # pyltp0.2.1应该是在这里改动
                if arcs[arc_index][1] in child_dict:
                    child_dict[arcs[arc_index][1]].append(arc_index)
                else:
                    child_dict[arcs[arc_index][1]] = []
                    child_dict[arcs[arc_index][1]].append(arc_index)
        child_dict_list.append(child_dict)
    return child_dict_list


def _bulid_connected_parse_dict(words, netags, arcs, index):
    _connected_parse_dict = {}
    _connected_parse_dict['parent'] = {}
    _connected_parse_dict['child'] = {}
    end_index = 0
    for netag_index in range(index, len(netags)):
        if netags[netag_index] in ['E-Ni', 'E-Nh']:
            end_index = netag_index
            break
        elif netags[netag_index] == 'O':
            end_index = netag_index - 1
        else:
            pass
    for conn_index in range(index, end_index+1):
        if arcs[conn_index] != 'ATT':
            _connected_parse_dict['parent'].setdefault(arcs[conn_index][1], [])
            _connected_parse_dict['parent'][arcs[conn_index][1]].append(arcs[conn_index][0] - 1)
    for arc_index, arc in enumerate(arcs):
        if arc[0] - 1 in [conn_index for conn_index in range(index, end_index+1)]:
            _connected_parse_dict['child'].setdefault(arc[1], [])
            _connected_parse_dict['child'][arc[1]].append(arc_index)
    return _connected_parse_dict


def _parse_fromSent(sent_info):
    """
    每个语法关系为一个元组储存 relation :(word1, word2, relation)
    """
    ###这里要加一个filterlist的通信###
    parent_relation = ['ATT', 'SBV']
    _relation = []
    words, postags, netags, arcs = sent_info
    child_dict_list = build_parse_child_dict(words, postags, arcs)  # 创建一个词语的子节点列表，方便后续使用
    isB_Entity = lambda x: x in ['B-Ni', 'B-Nh']
    for index, word in enumerate(words):
        if isB_Entity(netags[index]):
            _connected = _connect_entity(words, netags, index)
            if _connected in FILTER_LIST:
                _connected_parse_dict = _bulid_connected_parse_dict(words, netags, arcs, index)
                for relation in parent_relation:
                    if relation in _connected_parse_dict['parent']:
                        for parent_index in _connected_parse_dict['parent'][relation]:
                            _relation.append((_connected, words[parent_index], relation))
                if 'ATT' in _connected_parse_dict['child']:
                    for child_index in _connected_parse_dict['child']['ATT']:
                        _relation.append((_connected, words[child_index], 'ATT'))
        if word in FILTER_LIST:
            word_dict = child_dict_list[index]
            if arcs[index][1] in parent_relation:  # 提取该词做主语时的动词  注：这里的SBV和VOB只能选一边
                _relation.append((word, words[arcs[index][0] - 1], arcs[index][1]))
            if 'ATT' in word_dict:
                _att = [(word, words[i], 'ATT') for i in word_dict['ATT']]
                _relation.extend(_att)
    return _relation


def _parse_fromALL(WORDS, POSTAGS, NETAGS, ARCS, workers):
    global TARGET_DICT
    print('Start Extracting graph structure')
    t1 = time.time()
    with Pool(processes=workers) as _executor:
        relations = _executor.map(_parse_fromSent, zip(WORDS, POSTAGS, NETAGS, ARCS))
    print('Relation Extract finished, Time cost %f s' % (time.time() - t1))
    relations = flatten(relations)
    for item in relations:
        TARGET_DICT.setdefault(item[0], {})
        TARGET_DICT[item[0]].setdefault(item[1], 0)
        TARGET_DICT[item[0]][item[1]] += 1
    print('LIST2DICT finished, len of TARGET_DICT %d' % len(TARGET_DICT))


def delete_error(sent_list, drop_len=1900):
    sent_list = [sent if len(sent) <= drop_len else None for sent in sent_list]
    while None in sent_list:
        sent_list.remove(None)
    return sent_list


def start(contents, workers=WORKERS):
    '''
    主程序
    contents为一个可迭代类型，可以是Series/List
    '''
    global POTENTIAL_LIST, LABELED_LIST, FILTER_LIST, TARGET_DICT
    _get_labeled()
    SENTENCES = conts2sents(contents, workers)
    # 经过检查，这里需要剔除三个导致程序崩溃的句子 index = 115803   121315  739546，为了适用性，统一去除过长的句子。
    # 一般经过分句后，还是有1000多长度的句子可能是错句，有的会导致后续ltp过程崩溃
    SENTENCES = delete_error(SENTENCES)
    print('%d Sentences to deal' % len(SENTENCES))
    WORDS = get_words(SENTENCES, workers)
    POSTAGS = get_postags(WORDS, workers)
    NETAGS = get_netags(WORDS, POSTAGS, workers)
    ARCS = get_arcs(WORDS, POSTAGS, workers)
    _potentials = _findNE_fromALL(WORDS, POSTAGS, NETAGS, workers)
    POTENTIAL_LIST = flatten(_potentials)
    POTENTIAL_LIST = list(set(POTENTIAL_LIST))
    print('Len of POTENTIAL_LIST %d' % len(POTENTIAL_LIST))
    FILTER_LIST = list(set(POTENTIAL_LIST) | set(LABELED_LIST))
    print('Len of FILTER_LIST %d' % len(FILTER_LIST))
    _parse_fromALL(WORDS, POSTAGS, NETAGS, ARCS, workers)


def save_result():
    global POTENTIAL_LIST, FILTER_LIST, TARGET_DICT
    print('Saving result....')
    with open('Filter_list.json', 'w') as f:
        json.dump(FILTER_LIST, f)
    with open(POTENTIAL_FILE, 'w') as f:
        json.dump(POTENTIAL_LIST, f)
    with open('targetDict.json', 'w') as f:
        json.dump(TARGET_DICT, f)

    # 写入图文件
    gf = open(GRAPH_FILE, 'w', encoding='utf-8')
    sf = open(SEED_FILE, 'w', encoding='utf-8')
    pos = open('../datasets/merge_pos_dict.txt', 'r', encoding='utf-8')
    n_nodes = []
    for key1 in TARGET_DICT.keys():
        for key2 in TARGET_DICT[key1].keys():
            gf.write(key1 + '\t' + key2 + '\t' + str(1.0) + '\n')
            n_nodes.append(key1)
            n_nodes.append(key2)
    n_nodes = list(set(n_nodes))
    for line in pos.readlines():
        line = line.split()
        if (line[0] in n_nodes) & (line[1] in ['ni', 'nh']):
            sf.write(line[0] + '\t' + line[1] + '\t' + str(float(1.0)) + '\n')
    gf.close()
    sf.close()
    pos.close()


if __name__ == '__main__':
    t1 = time.time()
    DATA = pd.read_csv(INPUT_CSV_PATH, lineterminator='\n')
    DATA = text_clean.text_clean(DATA, _DOC_LENGTH_FFILTER=_DOC_LENGTH_FFILTER, content=CONTENT)
    contents = DATA[CONTENT]
    start(contents)
    save_result()
    MAD = maddl.ModifiedAdsorption(_tol=0.1, cores=21, sliec_n=1000, max_iter=10)
    MAD.start_MAD(GRAPH_FILE, SEED_FILE, POTENTIAL_FILE, OUTPUT_FILE)
    t2 = time.time()
    print('All Finished, the whole process cost %f s' % (t2 - t1))
