import pickle
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

evaluation_queries = [
    "termination clause",
    "payment terms",
    "confidential information",
    "governing law",
    "intellectual property",
    "indemnification",
    "limitation of liability",
    "force majeure",
    "assignment",
    "warranty",
    "royalty payment",
    "dispute resolution",
    "arbitration",
    "breach of contract",
    "notice period"
]


class ContractSearchEngine:


    def __init__(self,
                 model_path="models/word2vec_model/contract_w2v.model",
                 index_path="data/contract_index.pkl"):

        print("Loading model and index...")

        self.model = Word2Vec.load(model_path)

        with open(index_path, 'rb') as f:
            self.index = pickle.load(f)

        print(f"Loaded {len(self.index)} clauses\n")

        # Build TF-IDF on clause texts
        self.clause_texts = [item['clause_text'] for item in self.index]

        self.tfidf = TfidfVectorizer(
            stop_words='english',
            max_features=5000
        )

        self.tfidf_matrix = self.tfidf.fit_transform(self.clause_texts)



class ContractSearchEngine:

    def __init__(self,
                 model_path="models/word2vec_model/contract_w2v.model",
                 index_path="data/contract_index.pkl"):

        print("Loading model and index...")

        self.model = Word2Vec.load(model_path)

        with open(index_path, 'rb') as f:
            self.index = pickle.load(f)

        print(f"Loaded {len(self.index)} clauses\n")
        self.clause_texts = [item['clause_text'] for item in self.index]

        self.tfidf = TfidfVectorizer(
            stop_words='english',
            max_features=5000
        )

        self.tfidf_matrix = self.tfidf.fit_transform(self.clause_texts)

    def text_to_vector(self, text):
        words = text.lower().split()
        vectors = [self.model.wv[w] for w in words if w in self.model.wv]

        if vectors:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(self.model.wv.vector_size)

    # def search(self, query, top_k=5):
    #     query_vector = self.text_to_vector(query)

    #     results = []

    #     for item in self.index:
    #         score = cosine_similarity(
    #             query_vector.reshape(1, -1),
    #             item['clause_vector'].reshape(1, -1)
    #         )[0][0]

    #         results.append({
    #             'contract': item['contract_id'],
    #             'text': item['clause_text'],
    #             'score': score
    #         })

    #     results.sort(key=lambda x: x['score'], reverse=True)
    #     return results[:top_k]

    def search(self, query, top_k=5, alpha=0.7):

        # ---- Semantic Score ----
        query_vector = self.text_to_vector(query)

        semantic_scores = []
        for item in self.index:
            score = cosine_similarity(
                query_vector.reshape(1, -1),
                item['clause_vector'].reshape(1, -1)
            )[0][0]
            semantic_scores.append(score)

        semantic_scores = np.array(semantic_scores)

        # ---- TF-IDF Score ----
        query_tfidf = self.tfidf.transform([query])
        keyword_scores = cosine_similarity(query_tfidf, self.tfidf_matrix)[0]

        # ---- Hybrid Score ----
        final_scores = alpha * semantic_scores + (1 - alpha) * keyword_scores

        # ---- Collect Results ----
        results = []
        for i, item in enumerate(self.index):
            results.append({
                'contract': item['contract_id'],
                'text': item['clause_text'],
                'score': final_scores[i]
            })

        results.sort(key=lambda x: x['score'], reverse=True)

        return results[:top_k]

    
    def get_relevant_clauses(index, query):
        """
        Define relevance automatically:
        A clause is relevant if the query words appear in clause text.
        """

        relevant = []

        query_words = query.lower().split()

        for item in index:
            text = item['clause_text'].lower()

            if all(word in text for word in query_words):
                relevant.append(item['clause_id'])

        return relevant

def get_relevant_clauses(index, query):
    """
    Define relevance automatically:
    A clause is relevant if the query words appear in clause text.
    """

    relevant = []

    query_words = query.lower().split()

    for item in index:
        text = item['clause_text'].lower()

        if all(word in text for word in query_words):
            relevant.append(item['clause_id'])

    return relevant


    
def evaluate_engine(engine, queries, k=5):

        total_precision = 0
        total_recall = 0
        total_mrr = 0

        for query in queries:

            results = engine.search(query, top_k=k)
            relevant_clauses = get_relevant_clauses(engine.index, query)

            retrieved_ids = [
                r['contract'] + "_c" + r['text'][:1]  # dummy id workaround
                for r in results
            ]

            # Better: use clause_id directly if returned
            retrieved_ids = [r['contract'] for r in results]

            # Count relevant retrieved
            relevant_retrieved = 0
            first_relevant_rank = None

            for rank, r in enumerate(results, 1):

                clause_text = r['text'].lower()

                if all(word in clause_text for word in query.lower().split()):
                    relevant_retrieved += 1

                    if first_relevant_rank is None:
                        first_relevant_rank = rank

            precision = relevant_retrieved / k
            recall = (relevant_retrieved / len(relevant_clauses)) if relevant_clauses else 0

            total_precision += precision
            total_recall += recall

            if first_relevant_rank:
                total_mrr += 1 / first_relevant_rank

            print(f"\nQuery: {query}")
            print(f"Precision@{k}: {precision:.3f}")
            print(f"Recall@{k}: {recall:.3f}")

        n = len(queries)

        print("\n" + "="*50)
        print("FINAL METRICS")
        print("="*50)
        print(f"Average Precision@{k}: {total_precision/n:.3f}")
        print(f"Average Recall@{k}: {total_recall/n:.3f}")
        print(f"MRR: {total_mrr/n:.3f}")





if __name__ == "__main__":

    engine = ContractSearchEngine()
    query = "termination clause"

    results = engine.search(query)

    print(f"\nQuery: {query}\n")

    for i, r in enumerate(results, 1):
        print(f"{i}. {r['contract']} (score: {r['score']:.3f})")
        print(f"   {r['text']}\n")

    evaluate_engine(engine, evaluation_queries, k=5)