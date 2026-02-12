import pickle
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity


class ContractSearchEngine:

    def __init__(self,
                 model_path="models/word2vec_model/contract_w2v.model",
                 index_path="data/contract_index.pkl"):

        print("Loading model and index...")

        self.model = Word2Vec.load(model_path)

        with open(index_path, 'rb') as f:
            self.index = pickle.load(f)

        print(f"Loaded {len(self.index)} clauses\n")

    def text_to_vector(self, text):
        words = text.lower().split()
        vectors = [self.model.wv[w] for w in words if w in self.model.wv]

        if vectors:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(self.model.wv.vector_size)

    def search(self, query, top_k=5):
        query_vector = self.text_to_vector(query)

        results = []

        for item in self.index:
            score = cosine_similarity(
                query_vector.reshape(1, -1),
                item['clause_vector'].reshape(1, -1)
            )[0][0]

            results.append({
                'contract': item['contract_id'],
                'text': item['clause_text'],
                'score': score
            })

        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]


if __name__ == "__main__":

    engine = ContractSearchEngine()
    query = "termination clause"

    results = engine.search(query)

    print(f"\nQuery: {query}\n")

    for i, r in enumerate(results, 1):
        print(f"{i}. {r['contract']} (score: {r['score']:.3f})")
        print(f"   {r['text']}\n")
