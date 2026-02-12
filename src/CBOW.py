"""
STEP 3: Train Word2Vec Model
Trains the AI to understand contract language
"""

from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
import json
import os
from pathlib import Path
import time
import pickle

class TrainingCallback(CallbackAny2Vec):
    """
    Callback to show training progress
    """
    def __init__(self):
        self.epoch = 0
        self.losses = []
    
    def on_epoch_end(self, model):
        self.epoch += 1
        loss = model.get_latest_training_loss()
        self.losses.append(loss)
        
        if self.epoch % 5 == 0 or self.epoch == 1:
            print(f"      Epoch {self.epoch:2d}/30 complete")

def load_all_sentences(processed_folder='data/processed'):
    """
    Load all preprocessed sentences from JSON files
    """
    print("\nLoading preprocessed data...")
    
    all_sentences = []
    file_count = 0
    word_count = 0
    
    # Get all JSON files
    json_files = sorted([f for f in os.listdir(processed_folder) if f.endswith('.json')])
    
    if not json_files:
        print(f" ERROR: No JSON files found in {processed_folder}")
        print("   Run preprocessing first: python src/step2_preprocess.py")
        return None
    
    print(f"   Found {len(json_files)} processed contracts")
    
    for json_file in json_files:
        file_path = os.path.join(processed_folder, json_file)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                sentences = json.load(f)
                all_sentences.extend(sentences)
                file_count += 1
                
                # Count words
                for sentence in sentences:
                    word_count += len(sentence)
        except Exception as e:
            print(f"  Error loading {json_file}: {e}")
    
    print(f"\nLoaded successfully:")
    print(f"   Contracts: {file_count}")
    print(f"   Sentences: {len(all_sentences):,}")
    print(f"   Total words: {word_count:,}")
    print(f"   Avg words per sentence: {word_count/len(all_sentences):.1f}")
    
    return all_sentences

def analyze_vocabulary(sentences):
    """
    Analyze vocabulary before training
    """
    print("\nAnalyzing vocabulary...")
    
    # Get all unique words
    vocabulary = set()
    word_freq = {}
    
    for sentence in sentences:
        for word in sentence:
            vocabulary.add(word)
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Sort by frequency
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    
    print(f"   Unique words: {len(vocabulary):,}")
    print(f"   Most common words:")
    for word, count in sorted_words[:10]:
        print(f"      {word:20s}: {count:4d} times")
    
    # Check for legal terms
    legal_terms = [
        'payment_terms', 'net_30', 'net_60', 'delivery_term', 
        'liability', 'termination', 'agreement', 'contract',
        'force_majeure', 'indemnification'
    ]
    
    found_terms = [term for term in legal_terms if term in vocabulary]
    print(f"\n   Legal terms found: {len(found_terms)}/{len(legal_terms)}")
    if found_terms:
        print(f"   Examples: {', '.join(found_terms[:5])}")

def train_word2vec_model(sentences, save_path='models/word2vec_model'):
    """
    Train Word2Vec model on contract data
    """
    print("\n" + "="*70)
    print("TRAINING WORD2VEC MODEL")
    print("="*70 + "\n")
    
    print("Model Configuration:")
    print("   Vector Size:    200  (dimensionality of word embeddings)")
    print("   Window Size:    10   (context words to consider)")
    print("   Min Count:      2    (ignore rare words)")
    print("   Algorithm:      Skip-gram (better for small datasets)")
    print("   Epochs:         30   (training iterations)")
    print("   Workers:        4    (CPU cores)\n")
    
    print("Starting training...\n")
    
    start_time = time.time()
    
    # Initialize callback
    callback = TrainingCallback()
    
    # Train the model
    model = Word2Vec(
        sentences=sentences,
        vector_size=200,           # Size of word vectors
        window=5,                 # Context window (larger for legal docs)
        min_count=2,               # Ignore words appearing less than 2 times
        workers=4,                 # Number of CPU cores
        sg=0,                      # Skip-gram algorithm
        epochs=30,                 # Number of training epochs
        negative=15,               # Negative sampling
        ns_exponent=0.75,          # Negative sampling exponent
        alpha=0.025,               # Initial learning rate
        min_alpha=0.0001,          # Minimum learning rate
        seed=42,                   # For reproducibility
        compute_loss=True,         # Track training loss
        callbacks=[callback]       # Show progress
    )
    
    training_time = time.time() - start_time
    
    print(f"\nTraining complete!")
    print(f"   Time taken: {training_time:.1f} seconds ({training_time/60:.1f} minutes)")
    
    # Create save directory
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_file = os.path.join(save_path, "contract_w2v.model")
    model.save(model_file)
    print(f"\n Model saved: {model_file}")
    
    # Save vocabulary
    vocab = list(model.wv.index_to_key)
    vocab_file = os.path.join(save_path, "vocabulary.txt")
    with open(vocab_file, 'w', encoding='utf-8') as f:
        for word in vocab:
            f.write(f"{word}\n")
    print(f" Vocabulary saved: {vocab_file} ({len(vocab)} words)")
    
    # Save model info
    info = {
        'training_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'num_sentences': len(sentences),
        'vocabulary_size': len(vocab),
        'vector_size': 200,
        'window_size': 10,
        'min_count': 2,
        'training_time_seconds': training_time,
        'epochs': 30
    }
    
    info_file = os.path.join(save_path, "model_info.json")
    with open(info_file, 'w') as f:
        json.dump(info, f, indent=2)
    print(f"Model info saved: {info_file}")
    
    return model

def test_model(model):
    """
    Test the trained model with example words
    """
    print("\n" + "="*70)
    print("MODEL TESTING")
    print("="*70 + "\n")
    
    print("Testing word similarities:\n")
    
    # Test words (common contract terms)
    test_words = [
        'payment',
        'delivery',
        'liability',
        'termination',
        'agreement',
        'vendor',
        'net_30',
        'contract'
    ]
    
    tested_count = 0
    
    for word in test_words:
        if word in model.wv:
            print(f" Words similar to '{word}':")
            try:
                similar = model.wv.most_similar(word, topn=5)
                for sim_word, score in similar:
                    print(f"   {sim_word:25s} (similarity: {score:.3f})")
                print()
                tested_count += 1
            except:
                print(f"  Cannot compute similarity\n")
        else:
            print(f"'{word}' not in vocabulary\n")
    
    if tested_count == 0:
        print("No test words found in vocabulary")
        print("   This might indicate preprocessing issues")
        return
    
    # Test word arithmetic
    print("\n" + "="*70)
    print("WORD ARITHMETIC TESTS")
    print("="*70 + "\n")
    
    arithmetic_tests = [
        (['payment', 'terms'], [], "payment + terms"),
        (['delivery', 'schedule'], [], "delivery + schedule"),
        (['liability', 'limit'], [], "liability + limit"),
    ]
    
    for positive, negative, description in arithmetic_tests:
        # Check all words exist
        if all(word in model.wv for word in positive + negative):
            try:
                print(f"{description} =")
                result = model.wv.most_similar(positive=positive, negative=negative, topn=5)
                for word, score in result:
                    print(f"   {word:25s} (similarity: {score:.3f})")
                print()
            except:
                print(f"  Cannot compute\n")
    
    # Vocabulary statistics
    print("="*70)
    print("VOCABULARY STATISTICS")
    print("="*70 + "\n")
    
    vocab_size = len(model.wv)
    print(f"Total vocabulary size: {vocab_size:,} words")
    
    # Check for specific legal terms
    legal_terms = {
        'Payment Terms': ['payment', 'payment_terms', 'net_30', 'net_60', 'invoice', 'billing'],
        'Delivery Terms': ['delivery', 'delivery_term', 'shipment', 'shipping'],
        'Liability': ['liability', 'indemnification', 'damages', 'limit'],
        'Termination': ['termination', 'terminate', 'notice'],
        'Legal Concepts': ['force_majeure', 'governing_law', 'dispute_resolution']
    }
    
    print("\n📋 Legal Terms Coverage:\n")
    for category, terms in legal_terms.items():
        found = [t for t in terms if t in model.wv]
        print(f"   {category:20s}: {len(found)}/{len(terms)} found")
        if found:
            print(f"      Examples: {', '.join(found[:3])}")

def save_embeddings_for_visualization(model, save_path='models/CBOW_model'):
    """
    Save word embeddings in a format suitable for visualization
    """
    print("\nSaving embeddings for visualization...")
    
    embeddings_file = os.path.join(save_path, "embeddings.pkl")
    
    # Get top 500 most common words
    words = model.wv.index_to_key[:500]
    vectors = [model.wv[word] for word in words]
    
    with open(embeddings_file, 'wb') as f:
        pickle.dump({'words': words, 'vectors': vectors}, f)
    
    print(f"   Saved top 500 word embeddings: {embeddings_file}")

def main():
    """
    Main training pipeline
    """
    print("\n" + "="*70)
    print("WORD2VEC TRAINING PIPELINE")
    print("="*70)
    
    # Step 1: Load data
    sentences = load_all_sentences()
    
    if sentences is None:
        return
    
    # Check if we have enough data
    if len(sentences) < 500:
        print(f"\n WARNING: Only {len(sentences)} sentences found")
        print("   Recommended minimum: 1,000 sentences")
        print("   Model quality may be limited with small data")
        
        response = input("\n   Continue anyway? (yes/no): ").strip().lower()
        if response != 'yes':
            print("\nTraining cancelled")
            print("   Consider adding more contracts or checking preprocessing")
            return
    
    # Step 2: Analyze vocabulary
    analyze_vocabulary(sentences)
    
    # Step 3: Train model
    model = train_word2vec_model(sentences)
    
    # Step 4: Test model
    test_model(model)
    
    # Step 5: Save embeddings for visualization
    save_embeddings_for_visualization(model)
    
    # Final summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print("\nYour Word2Vec model is ready!")
    print(f"Model location: models/word2vec_model/contract_w2v.model")
    print(f"\nNext steps:")
    print(f"   1. Build search engine: python src/step4_build_search.py")
    print(f"   2. Add anomaly detection: python src/step5_anomaly_detection.py")
    print(f"   3. Create web interface: python src/step6_web_app.py")
    
    print("\n You can also test your model interactively:")
    print("   python src/test_model_interactive.py")

if __name__ == "__main__":
    main()
