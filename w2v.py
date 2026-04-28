from __future__ import print_function
from gensim.models import word2vec
from os.path import join, exists, split
import os
import numpy as np


def train_word2vec(sentence_matrix, vocabulary_inv,
                   num_features=300, min_word_count=1, context=10):
    """
    Trains, saves, loads Word2Vec model
    Returns initial weights for embedding layer.
    """
    model_dir = 'models'
    model_name = "{:d}features_{:d}minwords_{:d}context".format(num_features, min_word_count, context)
    model_name = join(model_dir, model_name)
    
    if exists(model_name):
        embedding_model = word2vec.Word2Vec.load(model_name)
        print('Load existing Word2Vec model \'%s\'' % split(model_name)[-1])
    else:
        num_workers = 2  
        downsampling = 1e-3  

        print('Training Word2Vec model...')
        sentences = [[vocabulary_inv[w] for w in s] for s in sentence_matrix]
        
        # PERBAIKAN 1: Ubah 'size' menjadi 'vector_size'
        embedding_model = word2vec.Word2Vec(sentences, workers=num_workers,
                                            vector_size=num_features, min_count=min_word_count,
                                            window=context, sample=downsampling)

        # PERBAIKAN 2: init_sims sudah dihapus di Gensim 4.x, kita matikan saja
        # embedding_model.init_sims(replace=True) 

        if not exists(model_dir):
            os.mkdir(model_dir)
        print('Saving Word2Vec model \'%s\'' % split(model_name)[-1])
        embedding_model.save(model_name)

    # PERBAIKAN 3: Gunakan enumerate() karena vocabulary_inv adalah List
    embedding_weights = {key: embedding_model.wv[word] if word in embedding_model.wv else
                              np.random.uniform(-0.25, 0.25, embedding_model.vector_size)
                         for key, word in enumerate(vocabulary_inv)}
    return embedding_weights

if __name__ == '__main__':
    import data_helpers

    print("Loading data...")
    x, _, _, vocabulary_inv_list = data_helpers.load_data()
    vocabulary_inv = {key: value for key, value in enumerate(vocabulary_inv_list)}
    w = train_word2vec(x, vocabulary_inv)
