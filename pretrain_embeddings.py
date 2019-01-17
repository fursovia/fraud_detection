"""
Pretrains embeddings for treatments
"""

from gensim.models import Word2Vec
import pandas as pd
from typing import List

DATA_PATH = 'data/only_treatments/full.csv'


def train_word2vec(documents: List[str], emb_dim: int, name: str):

    model = Word2Vec(sentences=documents, size=emb_dim, window=10, min_count=2, workers=10)

    model.train(documents, total_examples=len(documents), epochs=10)

    model.wv.save_word2vec_format(f'data/word2vec_{name}_{emb_dim}.txt', binary=False)


if __name__ == '__main__':

    data = pd.read_csv(DATA_PATH)

    treatments = data['treatments'].tolist()
    treatment_types = data['types'].tolist()

    sizes = [50, 100, 300]

    # TREATMENTS
    documents = [doc.split() for doc in treatments]

    for size in sizes:
        train_word2vec(documents=documents, emb_dim=size, name='treatments')

    # TREATMENT TYPES
    documents = [doc.split() for doc in treatment_types]

    for size in sizes:
        train_word2vec(documents=documents, emb_dim=size, name='types')
