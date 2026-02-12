import re
import os
import json

from pathlib import Path
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize ,word_tokenize

print("checking NLTK")

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("downdloading data")
    nltk.download('punkt',quiet=True)
    nltk.download('stopwords',quiet=True)

def remove_technical_artifacts(text):
    """
    Remove section numbers, legal citations, and technical artifacts
    """
    # Remove section numbers like "10.1a", "2.1bii", "3.00"
    text = re.sub(r'\b\d+\.\d+[a-z]*\b', '', text)
    
    # Remove standalone numbers with decimals
    text = re.sub(r'\b\d+\.\d+\b', '', text)
    
    # Remove legal citations like "Article 10", "Section 5.2"
    text = re.sub(r'\b(article|section|subsection|clause|paragraph)\s+\d+(\.\d+)?[a-z]?\b', '', text, flags=re.IGNORECASE)
    
    # Remove technical terms that aren't useful
    technical_noise = [
        'actual360', 'actual/360', 'denominator', 'numerator',
        'fractions', 'decimal', 'rounded', 'calculation', 'compute'
    ]
    
    for term in technical_noise:
        text = re.sub(r'\b' + term + r'\b', '', text, flags=re.IGNORECASE)
    
    return text
    

def clean_text(text):
    """
    Basic text cleaning
    """
    # Remove technical artifacts FIRST
    text = remove_technical_artifacts(text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep important punctuation
    text = re.sub(r'[^a-z0-9\s\.\,\-]', '', text)
    
    # Remove standalone numbers
    text = re.sub(r'\b\d+\b', '', text)
    
    return text.strip()

def handle_legal_phrases(text):
    """
    keep the multi owrd legal tersm together as single tokens
 Example: "force majeure" → "force_majeure"
    This helps Word2Vec understand they're a single concept
    

    """
    legal_terms = [
        # Common legal phrases
        'force majeure',
        'due diligence',
        'intellectual property',
        'non disclosure',
        'nondisclosure',
        'breach of contract',
        'governing law',
        'dispute resolution',
        'arbitration clause',
        'choice of law',
        
        # Payment terms
        'payment terms',
        'payment term',
        'payment schedule',
        'billing cycle',
        'invoice date',
        'due date',
        'net 30',
        'net 60',
        'net 90',
        'net thirty',
        'net sixty',
        'net ninety',
        
        # Delivery terms
        'delivery schedule',
        'delivery term',
        'shipping term',
        'delivery date',
        'business day',
        'business days',
        'working day',
        'working days',
        
        # Liability terms
        'liability clause',
        'limited liability',
        'liability cap',
        'liability limit',
        'indemnification clause',
        'hold harmless',
        
        # Termination terms
        'termination clause',
        'termination right',
        'termination notice',
        'early termination',
        'notice period',
        
        # Other common terms
        'confidentiality agreement',
        'service level agreement',
        'scope of work',
        'work product',
        'purchase order',
        'change order',
    ]
    
    for terms in legal_terms:
        text=text.replace(terms,terms.replace(' ','_'))

    return text

def preprocess_text(text):
    """
    Complete preprocessing pipeline
    
    Input: Raw text string
    Output: List of sentences, each sentence is a list of words
    
    Example:
    Input: "This is a CONTRACT. Payment terms are net 30 days."
    Output: [
        ['contract'],
        ['payment_terms', 'net_30', 'days']
    ]
    """
    text = clean_text(text)
    text = handle_legal_phrases(text)

    sentences = sent_tokenize(text)
    processed_sentences = []

    stop_words = set(stopwords.words('english'))

    keep_words = {
        'not', 'no', 'nor', 'without', 'against',
        'under', 'over', 'above', 'below', 'between',
        'shall', 'will', 'must', 'may'
    }

    stop_words = stop_words - keep_words

    for sent in sentences:
        words = word_tokenize(sent)

        words = [
            word for word in words
            if word not in stop_words
            and len(word) > 2
            and not word.isdigit()
        ]

        if len(words) >= 3:
            processed_sentences.append(words)

    return processed_sentences


def preprocess_single_file(input_path, output_path):
    """
    Preprocess one text file
    """
    filename = os.path.basename(input_path)
    
    # Read raw text
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except Exception as e:
        print(f" Error reading {filename}: {e}")
        return 0
    
    # Check if file has content
    if len(text) < 100:
        print(f" {filename} is too short ({len(text)} chars), skipping")
        return 0
    
    # Preprocess
    try:
        processed_sentences = preprocess_text(text)
    except Exception as e:
        print(f" Error processing {filename}: {e}")
        return 0
    
    # Save as JSON
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed_sentences, f, indent=2)
    except Exception as e:
        print(f"  Error saving {filename}: {e}")
        return 0
    
    return len(processed_sentences)

def preprocess_all_files(input_folder='data/extracted_text',
                         output_folder='data/processed'):
    """
    Preprocess all extracted text files
    """
    print("\n" + "="*70)
    print("STEP 2: TEXT PREPROCESSING")
    print("="*70 + "\n")
    
    # Create output folder
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Get all text files
    txt_files = [f for f in os.listdir(input_folder) if f.endswith('.txt')]
    
    if not txt_files:
        print(f" ERROR: No text files found in {input_folder}")
        print("   Make sure you've extracted PDFs first!")
        return
    
    print(f" Found {len(txt_files)} text files to preprocess\n")
    
    total_sentences = 0
    successful = 0
    failed = 0
    
    # Process each file
    for i, txt_file in enumerate(txt_files, 1):
        input_path = os.path.join(input_folder, txt_file)
        output_file = txt_file.replace('.txt', '.json')
        output_path = os.path.join(output_folder, output_file)
        
        print(f"[{i}/{len(txt_files)}] {txt_file[:510]}...")
        
        num_sentences = preprocess_single_file(input_path, output_path)
        
        if num_sentences > 0:
            print(f"      Created {num_sentences} sentences")
            total_sentences += num_sentences
            successful += 1
        else:
            failed += 1





            
    
    # Summary
    print("\n" + "="*70)
    print("PREPROCESSING COMPLETE")
    print("="*70)
    print(f"\n Successfully processed: {successful}/{len(txt_files)}")
    print(f" Failed: {failed}/{len(txt_files)}")
    print(f" Total sentences created: {total_sentences:,}")
    print(f" Average sentences per contract: {total_sentences//successful if successful > 0 else 0}")
    
    # Quality check
    avg_sentences = total_sentences / successful if successful > 0 else 0
    
    if avg_sentences < 50:
        print("\n WARNING: Low sentence count per contract")
        print("   This might indicate:")
        print("   - Contracts are very short")
        print("   - Extraction quality is poor")
        print("   - Too aggressive filtering")
    elif avg_sentences > 500:
        print("\n WARNING: Very high sentence count")
        print("   This might indicate:")
        print("   - Contracts include lots of boilerplate")
        print("   - Poor sentence segmentation")
    else:
        print("\nSentence count looks good!")
    
    # Show example
    if successful > 0:
        print("\n" + "="*70)
        print("EXAMPLE OUTPUT")
        print("="*70 + "\n")
        
        # Load first processed file
        first_json = os.path.join(output_folder, txt_files[0].replace('.txt', '.json'))
        
        try:
            with open(first_json, 'r') as f:
                example = json.load(f)
            
            print(f"First 5 sentences from {txt_files[0]}:\n")
            for i, sentence in enumerate(example[:5], 1):
                print(f"{i}. {' '.join(sentence)}")
            
            print(f"\n... and {len(example)-5} more sentences")
        except:
            pass
    
    print(f"\nOutput folder: {output_folder}")
    
    if total_sentences >= 1000:
        print("\nDATA READY FOR TRAINING!")
        print("Next step: python src/step3_train_word2vec.py")
    else:
        print("\n Limited data for training")
        print(f"   Current: {total_sentences} sentences")
        print(f"   Recommended: 1000+ sentences")
        print("\n   You can still proceed, but model quality may be limited.")
        print("   Consider adding more contracts if possible.")

def main():
    """
    Main function
    """
    preprocess_all_files()

if __name__ == "__main__":
    main()



    

