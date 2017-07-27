__author__ = 'PC-LiNing'

from gensim import corpora
import numpy as np
import redis
from gensim import corpora
from collections import defaultdict

word_embedding_size  = 200


def getRandom_vec():
    vec=np.random.rand(word_embedding_size)
    norm=np.sum(vec**2)**0.5
    normalized = vec / norm
    return normalized


def collect_word_embedding():
    # redis
    r = redis.StrictRedis(host='10.2.4.78', port=6379, db=0)
    count = 0
    data = np.zeros(shape=(28782, word_embedding_size), dtype=np.float32)
    dic = corpora.Dictionary.load('words.dict')
    for key,value in dic.items():
        result = r.get(value)
        if result is not None:
            vec = np.fromstring(result,dtype=np.float32)
            if vec.shape == (word_embedding_size,):
                count += 1
                print(value)
        else:
            vec = getRandom_vec()
        data[key] = vec
    print(count)
    np.save('vectors.npy',data)


collect_word_embedding()