"""
STEP 5: Build searchable contract index
Pre-processes all contracts for fast searching
"""

from gensim.models import Word2Vec
import numpy as np
import re
import json
import os
import pickle
from pathlib import Path

class ContractIndexBuilder:
    
    def __init__(self, model_path='models/word2vec_model/contract_w2v.model'):
        """Load trained Word2Vec model"""
        print("="*70)
        print("CONTRACT INDEX BUILDER")
        print("="*70 + "\n")
        print("Loading Word2Vec model...")
        
        self.model = Word2Vec.load(model_path)
        print(f"  Model loaded ({len(self.model.wv):,} words in vocabulary)\n")
        
    def text_to_vector(self, text):
        """
        Convert text to vector embedding
        
        How it works:
        1. Split text into words
        2. Get Word2Vec vector for each word
        3. Average all vectors together
        
        Example:
        text = "payment terms net thirty"
        vectors = [vec(payment), vec(terms), vec(net), vec(thirty)]
        result = average(vectors) → single vector of 200 numbers
        """
        words = text.lower().split()
        
        vectors = []
        for word in words:
            if word in self.model.wv:
                vectors.append(self.model.wv[word])
        
        if vectors:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(self.model.wv.vector_size)
    
    def split_into_clauses(self, sentences, sentences_per_clause=5):
        """
        Group sentences into searchable clauses
        
        Why: Contracts are long. We split them into ~5 sentence chunks
        so users get relevant snippets, not entire contracts.
        
        Example:
        Contract with 100 sentences → 20 clauses of 5 sentences each
        """
        clauses = []
        
        for i in range(0, len(sentences), sentences_per_clause):
            chunk = sentences[i:i + sentences_per_clause]
            
            # Combine sentences into clause text
            clause_text = ' '.join([' '.join(sent) for sent in chunk])
            
            # Only keep substantial clauses (at least 10 words)
            if len(clause_text.split()) >= 10:
                clauses.append({
                    'text': clause_text,
                    'start_idx': i,
                    'end_idx': i + len(chunk)
                })
        
        return clauses
    
    def build_index(self, processed_folder='data/processed', 
                raw_folder='data/extracted_text'):

        print("="*70)
        print("BUILDING INDEX")
        print("="*70 + "\n")

        index = []

        json_files = sorted([
            f for f in os.listdir(processed_folder)
            if f.endswith('.json')
        ])

        print(f"📊 Found {len(json_files)} contracts to index\n")

        total_clauses = 0

        for file_count, json_file in enumerate(json_files, 1):

            contract_id = json_file.replace('.json', '')
            processed_path = os.path.join(processed_folder, json_file)
            raw_path = os.path.join(raw_folder, contract_id + ".txt")

            # Load processed sentences
            try:
                with open(processed_path, 'r', encoding='utf-8') as f:
                    sentences = json.load(f)
            except Exception as e:
                print(f" Error loading {json_file}: {e}")
                continue

            if len(sentences) < 5:
                continue

            # Load raw full text
            try:
                with open(raw_path, 'r', encoding='utf-8') as f:
                    raw_text = f.read()
            except:
                raw_text = ""

            
            raw_sentences = re.split(r'(?<=[.!?])\s+', raw_text)

            # Create clauses from processed text
            for clause_idx in range(0, len(sentences), 5):

                processed_chunk = sentences[clause_idx:clause_idx + 5]
                raw_chunk = raw_sentences[clause_idx:clause_idx + 5]

                processed_text = ' '.join([' '.join(sent) for sent in processed_chunk])
                raw_text_chunk = ' '.join(raw_chunk)

                if len(processed_text.split()) < 10:
                    continue

                clause_vector = self.text_to_vector(processed_text)

                index.append({
                    'contract_id': contract_id,
                    'clause_id': f"{contract_id}_c{clause_idx}",
                    'clause_text': raw_text_chunk,  # only 5 raw sentences
                    'clause_vector': clause_vector
                })


                total_clauses += 1

            if file_count % 50 == 0:
                print(f"[{file_count}/{len(json_files)}] Indexed {total_clauses:,} clauses...")

        print(f"\nIndexing complete!")
        print(f"Contracts: {len(json_files)}")
        print(f"Total clauses: {total_clauses:,}")

        return index

    
    def save_index(self, index, output_file='data/contract_index.pkl'):
        """Save index to disk"""
        print(f"\nSaving index to disk...")
        
        Path(os.path.dirname(output_file)).mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'wb') as f:
            pickle.dump(index, f)
        
        size_mb = os.path.getsize(output_file) / (1024 * 1024)
        print(f"  Saved: {output_file} ({size_mb:.1f} MB)")
    
    def test_search(self, index):
        """Quick test of the index"""
        from sklearn.metrics.pairwise import cosine_similarity
        
        print("\n" + "="*70)
        print("TESTING INDEX")
        print("="*70 + "\n")
        
        test_query = "termination"
        print(f"🔍 Test query: '{test_query}'\n")
        
        # Convert query to vector
        query_vector = self.text_to_vector(test_query)
        
        # Find similar clauses
        results = []
        for item in index:
            similarity = cosine_similarity(
                query_vector.reshape(1, -1),
                item['clause_vector'].reshape(1, -1)
            )[0][0]
            
            results.append({
                'contract': item['contract_id'],
                'text': item['clause_text'],
                'score': similarity
            })
        
        # Sort and show top 3
        results.sort(key=lambda x: x['score'], reverse=True)
        
        print("📋 Top 3 results:\n")
        for i, r in enumerate(results[:10], 1):
            print(f"{i}. {r['contract']} (score: {r['score']:.3f})")
            print(f"   {r['text']}\n")

def main():
    # Step 1: Initialize
    builder = ContractIndexBuilder()
    
    # Step 2: Build index
    index = builder.build_index()
    
    # Step 3: Save index
    builder.save_index(index)
    
    # Step 4: Test
    builder.test_search(index)
    
    print("\n" + "="*70)
    print("INDEX READY!")
    print("="*70)
    print("\nSearch index created successfully!")
    print("Location: data/contract_index.pkl")
    print("\n Next step:")
    print("   python src/step6_search_engine.py")

if __name__ == "__main__":
    main()
