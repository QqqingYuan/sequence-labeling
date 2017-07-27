__author__ = 'PC-LiNing'

import codecs
from gensim import corpora
from collections import  defaultdict
import numpy

# valid 3104
# test 3327
# train 13447
# all 19878
# length = [2,40]
# label = 9
# words = 28782


def load_corpus(file):
    corpus = codecs.open(file,'r',encoding='utf8')
    sentences = []
    sentence = []
    for line in corpus.readlines():
        line = line.strip('\n').strip()
        if line == "-DOCSTART- -X- -X- O":
            continue
        if line is not "":
            fields = line.split(' ')
            pair = (fields[0], fields[3])
            sentence.append(pair)
        else:
            if len(sentence) > 1 and len(sentence) <= 40:
                sentences.append(sentence.copy())
            sentence.clear()
    return sentences


def get_sent(sentence):
    return [item[0] for item in sentence]


def get_label(sentence):
    return [item[1] for item in sentence]


def print_lengths(lengths):
    temp = set(lengths)
    for item in temp:
        print(str(item)+": "+str(lengths.count(item)))


def generate_dic():
    train_sents = load_corpus('CoNLL-2003/train.txt')
    valid_sents = load_corpus('CoNLL-2003/valid.txt')
    test_sents = load_corpus('CoNLL-2003/test.txt')
    train_ = [get_sent(sent) for sent in train_sents]
    print("train size: "+str(len(train_sents)))
    valid_ = [get_sent(sent) for sent in valid_sents]
    print("valid size: "+str(len(valid_sents)))
    test_ = [get_sent(sent) for sent in test_sents]
    print("test size: "+str(len(test_sents)))
    all_ = train_ + valid_ + test_
    lengths = [len(text) for text in all_]
    print("all data: "+str(len(lengths)))
    print_lengths(lengths)
    dic_words = corpora.Dictionary(all_)
    dic_words.save('words.dict')
    print(len(dic_words))
    # label
    train_.clear()
    valid_.clear()
    test_.clear()
    train_ = [get_label(sent) for sent in train_sents]
    valid_ = [get_label(sent) for sent in valid_sents]
    test_ = [get_label(sent) for sent in test_sents]
    all_ = train_ + valid_ + test_
    dic_labels = corpora.Dictionary(all_)
    for key,value in dic_labels.items():
        print(value)
    print(len(dic_labels))


def invert_dict(d):
    return dict([(v,k) for k,v in d.items()])

label_str = {
    'B-ORG':0,
    'B-LOC':1,
    'B-PER':2,
    'B-MISC':3,
    'I-ORG':4,
    'I-MISC':5,
    'I-PER':6,
    'I-LOC':7,
    'O':8
}


def transfer_corpus(sents):
    words_dict = invert_dict(corpora.Dictionary.load('words.dict'))
    max_length = 40
    sentence = numpy.zeros(shape=(len(sents), max_length),dtype=numpy.int32)
    label = numpy.zeros(shape=(len(sents), max_length), dtype=numpy.int32)
    lengths = []
    for i in range(len(sents)):
        current_sent = sents[i]
        words = []
        labels = []
        lengths.append(len(current_sent))
        for item in current_sent:
            words.append(words_dict[item[0]])
            labels.append(label_str[item[1]])
        sentence[i] = numpy.asarray(words + (max_length - len(current_sent))*[28782],dtype=numpy.float32)
        label[i] = numpy.asarray(labels + (max_length - len(current_sent))*[8],dtype=numpy.float32)

    return sentence,label,numpy.asarray(lengths,dtype=numpy.int32)


# train = train_ + valid_ = 16551
# test = test = 3327
def load_train_test():
    train_sents = load_corpus('CoNLL-2003/train.txt')
    valid_sents = load_corpus('CoNLL-2003/valid.txt')
    test_sents = load_corpus('CoNLL-2003/test.txt')
    train_ = train_sents + valid_sents
    test_ = test_sents
    train_sent,train_label,train_length = transfer_corpus(train_)
    test_sent, test_label, test_length = transfer_corpus(test_)
    return train_sent,train_label,train_length,test_sent,test_label,test_length




