# topic_model_pipeline.py

import os
from dotenv import load_dotenv
import torch
from bertopic import BERTopic
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from gensim.utils import simple_preprocess
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.datasets import fetch_20newsgroups
import argparse
import time
import gc

# Load environment variables from .env file in the experiments directory
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

# Get Hugging Face token from environment variable
HF_TOKEN = os.getenv('HUGGING_FACE_HUB_TOKEN')
if HF_TOKEN:
    os.environ['HUGGING_FACE_HUB_TOKEN'] = HF_TOKEN
    print("Hugging Face token loaded successfully")
else:
    print("Warning: HUGGING_FACE_HUB_TOKEN not found in environment variables")

# Set device and memory management
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def clear_gpu_memory():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

class HFTransformerEmbedder:
    def __init__(self, model_name, device="cuda", batch_size=8, use_quantization=True):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name,
            device_map=device,
            torch_dtype=torch.float16,  # Use FP16 by default
            load_in_8bit=use_quantization,  # Enable 8-bit quantization
            low_cpu_mem_usage=True
        )
        self.batch_size = batch_size
        self.device = device

    def encode(self, texts, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
            
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding texts"):
            batch_texts = texts[i:i + batch_size]
            inputs = self.tokenizer(batch_texts, padding=True, truncation=True, 
                                  max_length=512, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use mean pooling
                batch_embeddings = outputs.last_hidden_state.mean(dim=1)
                embeddings.append(batch_embeddings.cpu().numpy())
                
        return np.vstack(embeddings)

def calculate_topic_coherence(texts, topic_words, dictionary, corpus):
    """Calculate topic coherence using gensim's CoherenceModel."""
    coherence_model = CoherenceModel(
        topics=topic_words,
        texts=texts,
        dictionary=dictionary,
        corpus=corpus,
        coherence='c_v'
    )
    return coherence_model.get_coherence()

def calculate_topic_diversity(topic_words):
    """Calculate topic diversity as the average pairwise cosine similarity between topics."""
    # Convert topic words to a single string for each topic
    topic_texts = [' '.join(words) for words in topic_words]
    
    # Create a simple TF-IDF vectorizer
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer()
    topic_vectors = vectorizer.fit_transform(topic_texts)
    
    # Calculate pairwise cosine similarities
    similarities = cosine_similarity(topic_vectors)
    
    # Get the average similarity (excluding self-similarities)
    np.fill_diagonal(similarities, 0)
    avg_similarity = np.mean(similarities)
    
    # Convert to diversity score (1 - similarity)
    diversity = 1 - avg_similarity
    return diversity

def evaluate_model(texts, encoder):
    name, enc_type = encoder['name'], encoder['type']
    params = encoder.get('params', None)
    use_quantization = encoder.get('use_quantization', False)
    batch_size = encoder.get('batch_size', 8)
    
    print(f"\nRunning BERTopic with encoder: {name}")
    start_time = time.time()
    
    try:
        if enc_type == "sentence-transformer":
            embedder = SentenceTransformer(name, device=device)
        elif enc_type == "huggingface":
            embedder = HFTransformerEmbedder(
                name, 
                device=device, 
                batch_size=batch_size,
                use_quantization=use_quantization
            ).encode
        else:
            return None

        # Clear GPU memory before BERTopic
        clear_gpu_memory()
        
        topic_model = BERTopic(
            embedding_model=embedder,
            verbose=True,
            calculate_probabilities=True,
            n_gram_range=(1, 2)
        )
        topics, _ = topic_model.fit_transform(texts)

        # Prepare data for coherence calculation
        processed_texts = [simple_preprocess(text) for text in texts]
        dictionary = Dictionary(processed_texts)
        corpus = [dictionary.doc2bow(text) for text in processed_texts]
        
        # Get topic words from the model
        topic_words = [[word for word, _ in words] for words in topic_model.get_topics().values()]
        
        # Calculate metrics
        coherence = calculate_topic_coherence(processed_texts, topic_words, dictionary, corpus)
        diversity = calculate_topic_diversity(topic_words)

        elapsed_time = time.time() - start_time

        result = {
            "encoder": name,
            "type": enc_type,
            "params": params,
            "use_quantization": use_quantization,
            "batch_size": batch_size,
            "coherence": coherence,
            "diversity": diversity,
            "num_topics": len(set(topics)) - (1 if -1 in topics else 0),
            "running_time_sec": elapsed_time
        }

        # Save model
        model_path = f"models/bertopic_{name.replace('/', '_')}"
        topic_model.save(model_path)
        print(f"Model saved to {model_path}")
        
        # Clear memory after processing
        clear_gpu_memory()
        
        return result
        
    except Exception as e:
        print(f"Error processing model {name}: {str(e)}")
        return None

def run_pipeline(texts, encoders):
    results = []
    for encoder in tqdm(encoders, desc="Evaluating encoders", unit="encoder"):
        # Skip Llama 2 models by default to avoid OOM errors
        #if encoder["name"].startswith("meta-llama/"):
        #    print(f"Skipping {encoder['name']} due to GPU memory constraints. Remove this check to enable.")
        #    continue
        result = evaluate_model(texts, encoder)
        if result is not None:
            results.append(result)
    return pd.DataFrame(results)

def load_dummy_dataset():
    """Load the dummy dataset from file."""
    with open(os.path.join(os.path.dirname(__file__), 'dummy_dataset.txt'), 'r') as f:
        return [line.strip() for line in f if line.strip()]

def load_20newsgroups(subset='all', categories=None, remove_headers=True):
    """Load the 20 newsgroups dataset."""
    newsgroups = fetch_20newsgroups(
        subset=subset,
        categories=categories,
        remove=('headers',) if remove_headers else None,
        random_state=42
    )
    return newsgroups.data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run topic modeling pipeline on different datasets')
    parser.add_argument('--dataset', type=str, choices=['dummy', '20newsgroups'], default='dummy',
                      help='Dataset to use (dummy or 20newsgroups)')
    parser.add_argument('--subset', type=str, choices=['train', 'test', 'all'], default='all',
                      help='Subset of 20newsgroups to use (only applicable for 20newsgroups dataset)')
    parser.add_argument('--remove-headers', action='store_true',
                      help='Remove headers from 20newsgroups data')
    parser.add_argument('--categories', type=str, nargs='+',
                      help='Categories to use from 20newsgroups (e.g., alt.atheism comp.graphics)')
    
    args = parser.parse_args()
    
    # Load the appropriate dataset
    if args.dataset == 'dummy':
        texts = load_dummy_dataset()
        print(f"Loaded dummy dataset with {len(texts)} texts")
    else:  # 20newsgroups
        texts = load_20newsgroups(
            subset=args.subset,
            categories=args.categories,
            remove_headers=args.remove_headers
        )
        print(f"Loaded 20newsgroups dataset with {len(texts)} texts")
        if args.categories:
            print(f"Using categories: {args.categories}")

    encoders = [
        {
            "name": "all-MiniLM-L6-v2",
            "type": "sentence-transformer",
            "params": 22000000,
            "description": "all-MiniLM-L6-v2: 22M parameters, Sentence-Transformer. Download: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2"
        },
        {
            "name": "distilbert-base-uncased",
            "type": "huggingface",
            "params": 66000000,
            "description": "DistilBERT: 66M parameters, Hugging Face Transformers. Download: https://huggingface.co/distilbert-base-uncased"
        },
        {
            "name": "bert-base-uncased",
            "type": "huggingface",
            "params": 110000000,
            "description": "BERT: 110M parameters, Hugging Face Transformers. Download: https://huggingface.co/bert-base-uncased"
        },
        {
            "name": "roberta-base",
            "type": "huggingface",
            "params": 125000000,
            "description": "RoBERTa: 125M parameters, Hugging Face Transformers. Download: https://huggingface.co/roberta-base"
        },
        {
            "name": "microsoft/MiniLM-L12-H384-uncased",
            "type": "huggingface",
            "params": 33000000,
            "description": "MiniLM: 33M parameters, Hugging Face Transformers. Download: https://huggingface.co/microsoft/MiniLM-L12-H384-uncased"
        },
        {
            "name": "meta-llama/Llama-2-7b-hf",
            "type": "huggingface",
            "params": 7e9,  # 7B parameters
            "use_quantization": True,  # Enable 8-bit quantization
            "batch_size": 2  # Smaller batch size for memory efficiency
        },
        {
            "name": "meta-llama/Llama-2-13b-hf",
            "type": "huggingface",
            "params": 13000000000,
            "description": "Llama 2 13B: 13B parameters, Hugging Face Transformers. Download: https://huggingface.co/meta-llama/Llama-2-13b-hf"
        }
    ]

    os.makedirs("models", exist_ok=True)
    df_results = run_pipeline(texts, encoders)
    df_results.to_csv("results.csv", index=False)
    print("\nPipeline finished. Results saved to results.csv")
