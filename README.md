LexiSearch – Intelligent Legal Clause Retrieval System

LexiSearch is an NLP-based legal contract search engine that retrieves semantically relevant contract clauses using vector similarity techniques. Instead of simple keyword matching, it understands the meaning of queries and finds the most contextually similar clauses from a contract corpus.

This project demonstrates practical applications of:

Word Embeddings

Semantic Search

Information Retrieval Evaluation

Contract Clause Indexing

Features

Semantic clause search (beyond keyword matching)

Clause-level contract indexing

Word2Vec-based embeddings

Retrieval evaluation using Precision@K and Recall@K

Efficient similarity-based ranking

Tested on real contract datasets

Project Architecture
Contracts Dataset
        ↓
Text Preprocessing
        ↓
Clause Segmentation
        ↓
Word Embedding Generation (Word2Vec)
        ↓
Clause Vector Representation
        ↓
Similarity Search (Cosine Similarity)
        ↓
Ranked Results (Top-K Clauses)

Technologies Used

Python

NumPy

Pandas

Scikit-learn

Gensim (Word2Vec)

NLTK

Pickle (for index storage)

Project Structure
LexiSearch/
│
├── data/                  # Contracts dataset
├── models/                # Word2Vec model files
├── src/
│   ├── contract.py        # Contract index builder
│   ├── search.py          # Query & similarity search
│   ├── evaluation.py      # Precision@K, Recall@K metrics
│   └── preprocessing.py   # Text cleaning & tokenization
│
├── results/               # Output results
├── requirements.txt
└── README.md

How It Works
1. Preprocessing

Lowercasing

Tokenization

Stopword removal

Stemming (Porter Stemmer)

2. Word Embeddings

The system uses Word2Vec to generate dense vector representations for words. Clause embeddings are created by averaging word vectors.

3. Similarity Matching

The query is converted into vector form

Cosine similarity is calculated against indexed clauses

The top-K most similar clauses are returned

Evaluation Metrics

The system is evaluated using:

Precision@K

Recall@K

These metrics measure how well the model retrieves relevant clauses for a given query.

Example Queries

"termination clause"

"payment terms"

"confidentiality agreement"

"liability limitation"

The system retrieves semantically similar clauses even if the exact keywords are not present.

Installation
git clone https://github.com/sri15soumya/LexiSearch.git
cd LexiSearch
pip install -r requirements.txt

Usage
Build Index
python src/contract.py

Run Search
python src/search.py

Run Evaluation
python src/evaluation.py

Key Learnings

Understanding the difference between keyword search and semantic search

Practical implementation of Word2Vec

Building a complete information retrieval pipeline

Evaluating search systems using standard metrics

Efficient handling of large contract datasets

Future Improvements

Replace Word2Vec with contextual embeddings (transformer-based models)

Implement hybrid ranking with BM25

Deploy as a web application (Flask / FastAPI)

Develop a user interface for legal professionals

Improve ranking with fine-tuned embeddings
