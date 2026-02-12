"""
Interactive model testing
"""

from gensim.models import Word2Vec
import os

def test_interactive():
    """
    Interactive testing of Word2Vec model
    """
    model_path = 'models/word2vec_model/contract_w2v.model'
    
    if not os.path.exists(model_path):
        print("❌ Model not found. Train it first!")
        print("   Run: python src/step3_train_word2vec.py")
        return
    
    print("Loading model...")
    model = Word2Vec.load(model_path)
    
    print("\n" + "="*70)
    print("INTERACTIVE WORD2VEC TESTING")
    print("="*70)
    print("\nCommands:")
    print("  similar <word>        - Find similar words")
    print("  add <word1> <word2>   - Word arithmetic (word1 + word2)")
    print("  vocab                 - Show vocabulary size")
    print("  quit                  - Exit")
    print("\nExamples:")
    print("  similar payment")
    print("  add payment terms")
    print("="*70 + "\n")
    
    while True:
        try:
            command = input(">>> ").strip().lower()
            
            if command == 'quit':
                print("Goodbye!")
                break
            
            elif command == 'vocab':
                print(f"Vocabulary size: {len(model.wv):,} words")
            
            elif command.startswith('similar '):
                word = command.split(' ', 1)[1]
                
                if word in model.wv:
                    similar = model.wv.most_similar(word, topn=10)
                    print(f"\nWords similar to '{word}':")
                    for sim_word, score in similar:
                        print(f"  {sim_word:30s} {score:.3f}")
                else:
                    print(f"❌ '{word}' not in vocabulary")
            
            elif command.startswith('add '):
                parts = command.split()[1:]
                if len(parts) >= 2:
                    word1, word2 = parts[0], parts[1]
                    
                    if word1 in model.wv and word2 in model.wv:
                        result = model.wv.most_similar(positive=[word1, word2], topn=10)
                        print(f"\n{word1} + {word2} =")
                        for word, score in result:
                            print(f"  {word:30s} {score:.3f}")
                    else:
                        if word1 not in model.wv:
                            print(f"❌ '{word1}' not in vocabulary")
                        if word2 not in model.wv:
                            print(f"❌ '{word2}' not in vocabulary")
                else:
                    print("Usage: add <word1> <word2>")
            
            else:
                print("Unknown command. Type 'quit' to exit.")
            
            print()
        
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    test_interactive()