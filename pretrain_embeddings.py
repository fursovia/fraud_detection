"""
Pretrains embeddings for treatments
"""

from gensim.models import Word2Vec
import pandas as pd
from typing import List
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='data/test/full.csv')
parser.add_argument('--window', type=int, default=40)


def train_word2vec(documents: List[str], emb_dim: int, name: str, window_size=40):

    model = Word2Vec(sentences=documents, size=emb_dim, window=window_size, min_count=2, workers=10, negative=10)

    model.train(documents, total_examples=len(documents), epochs=30)

    model.wv.save_word2vec_format(f'data/word2vec_{name}_{emb_dim}.txt', binary=False)


if __name__ == '__main__':
    args = parser.parse_args()
    data = pd.read_csv(args.data_path)

    treatments = data['treatments'].tolist()
    treatment_types = data['types'].tolist()

    sizes = [50, 100, 300]

    # TREATMENTS
    documents = [doc.split() for doc in treatments]

    for size in sizes:
        train_word2vec(documents=documents, emb_dim=size, name='treatments', window_size=args.window)

    # # TREATMENT TYPES
    # documents = [doc.split() for doc in treatment_types]
    #
    # for size in sizes:
    #     train_word2vec(documents=documents, emb_dim=size, name='types', window_size=args.window)
