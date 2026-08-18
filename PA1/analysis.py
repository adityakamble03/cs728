import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

def get_top_n_neighbors(target_word, vocab, embedding_matrix, n=5):
    if target_word not in vocab:
        return f"'{target_word}' not found in vocabulary."
    
    # 1. Get the vector for the target word
    word_idx = vocab[target_word]
    target_vector = embedding_matrix[word_idx].reshape(1, -1)
    
    # 2. Calculate cosine similarity between target and ALL words
    # Result is a 1D array of similarity scores
    similarities = cosine_similarity(target_vector, embedding_matrix).flatten()
    
    # 3. Get indices of top scores (excluding the word itself at index 0)
    # argsort sorts ascending, so we take the last n+1 and reverse them
    related_indices = similarities.argsort()[-(n+1):][::-1]
    
    # 4. Map indices back to words
    id_to_word = {i: w for w, i in vocab.items()}
    
    results = []
    for idx in related_indices:
        word = id_to_word[idx]
        if word != target_word: # Skip the word itself
            results.append((word, similarities[idx]))
            
    return results[:n]

# --- EXECUTION ---
# Load your data

def solve(emb_sz = "200"):
    with open('embeddings/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)

    # Load one of your SVD matrices (e.g., k=200)
    W_svd = np.load(f'embeddings/W_svd_{emb_sz}.npy')

    targets = ["government", "economy", "technology"]

    print(f"Top 5 Nearest Neighbors (SVD k={emb_sz}):")
    print("-" * 40)
    for target in targets:
        neighbors = get_top_n_neighbors(target, vocab, W_svd)
        print(f"\nTarget: {target}")
        if isinstance(neighbors, list):
            for i, (word, score) in enumerate(neighbors, 1):
                print(f"  {i}. {word} (Score: {score:.4f})")
        else:
            print(f"  {neighbors}")

ls = ["50","100","200","300"]

for emb_sz in ls:
    print(f"\n\n--- Analyzing SVD with k={emb_sz} ---")
    solve(emb_sz)

