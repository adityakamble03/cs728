import json
import numpy as np
import os
import pickle
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer

def build_from_json(json_path, k_values=[50, 100, 200, 300]):
    print(f"Loading JSON data from {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data_dict = json.load(f)

    # 1. Map vocabulary to indices and identify total documents
    # vocab_list: the keys from the JSON
    vocab_list = list(data_dict.keys())
    print("THis is vocab list:", vocab_list[:5])
    word_to_id = {word: i for i, word in enumerate(vocab_list)}
    vocab_size = len(vocab_list)
    
    # Identify the number of documents (D)
    # The assignment says ~67k documents. We find the max index in the lists.
    max_doc_idx = 0
    for word in data_dict:
        for entry in data_dict[word]:
            # Assuming entry format is [doc_index, passage_text]
            doc_idx = entry[0]
            if doc_idx > max_doc_idx:
                max_doc_idx = doc_idx
    
    num_docs = max_doc_idx + 1 
    print(f"Vocab size: {vocab_size}, Total Documents: {num_docs}")

    # 2. Build Sparse Matrix (V x D)
    print("Populating sparse matrix...")
    rows = [] # Word IDs
    cols = [] # Doc IDs
    values = [] # Counts

    for word, entries in data_dict.items():
        word_id = word_to_id[word]
        for doc_idx, passage in entries:
            # We need to count occurrences of 'word' in 'passage'
            # Note: The assignment says Case Sensitive!
            count = passage.count(word) 
            if count > 0:
                rows.append(word_id)
                cols.append(doc_idx)
                values.append(count)

    X = csr_matrix((values, (rows, cols)), shape=(vocab_size, num_docs))
    
    # 3. Apply TF-IDF (Treated as Docs x Vocab for sklearn)
    print("Applying TF-IDF...")
    tfidf = TfidfTransformer()
    X_tfidf = tfidf.fit_transform(X.T).T

    # 4. SVD and Saving
    if not os.path.exists('embeddings'): os.makedirs('embeddings')
    
    with open('embeddings/vocab.pkl', 'wb') as f:
        pickle.dump(word_to_id, f)

    for k in k_values:
        print(f"Running SVD for k={k}...")
        svd = TruncatedSVD(n_components=k, algorithm='arpack', random_state=42)
        W_svd = svd.fit_transform(X_tfidf)
        
        np.save(f"embeddings/W_svd_{k}.npy", W_svd)
        print(f"Saved k={k} matrix.")

if __name__ == "__main__":
    # Update with your actual filename
    build_from_json("updated_vocab_document_dict.json")


# import json

# def preview_json(file_path, num_keys=3):
#     with open('updated_vocab_document_dict.json', 'r', encoding='utf-8') as f:
#         # We use a trick: load the data but only look at a slice
#         # Note: For a 1.1GB file, json.load might still take ~2-4GB of RAM
#         data = json.load(f) 
        
#         print(f"Total words in vocabulary: {len(data)}")
#         for i, (word, entries) in enumerate(data.items()):
#             if i >= num_keys: break
#             print(f"\nWord: '{word}'")
#             print(f"Number of occurrences: {len(entries)}")
#             print(f"First occurrence (Doc ID, Text): {entries[0]}")

# preview_json('updated_vocab_document_dict.json')

# data.keys()
