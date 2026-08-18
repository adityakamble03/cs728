import json
import os
import warnings
import logging

import json
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util

def load_data():
  with open('data/tools.json', 'r') as f:
    tools_data = json.load(f)
    tools_names = list(tools_data.keys())
    tool_descriptions = list(tools_data.values())

  with open('data/test_queries.json', 'r') as f:
    queries_data = json.load(f)

  return tools_names, tool_descriptions, queries_data

if __name__ == '__main__':
  tool_names, tool_descriptions, queries_data = load_data()
  print(f"Loaded {len(tool_names)} tools and {len(queries_data)} queries.")
  print(f"FIrst tool: {tool_names[0]} -> {tool_descriptions[0][:50]}...")

def evaluate_recall(predictions, queries_data):
  hits_at_1 = 0
  hits_at_5 = 0

  for i, query in enumerate(queries_data):
    gold_tool = query['gold_tool_name']

    top_1_guess = predictions[i][:1]
    top_5_guesses = predictions[i][:5]

    if gold_tool in top_1_guess:
      hits_at_1 += 1
    if gold_tool in top_5_guesses:
      hits_at_5 += 1

  recall_at_1 = hits_at_1 / len(queries_data)
  recall_at_5 = hits_at_5 / len(queries_data)

  return recall_at_1, recall_at_5

# BM25 Baseline
def evaluate_bm25(tool_names, tool_descriptions, queries_data):
  tokenized_tools = [desc.lower().split() for desc in tool_descriptions]

  bm25 = BM25Okapi(tokenized_tools)
  predictions = []

  for q in queries_data:
    tokenized_query = q['text'].lower().split()
    scores = bm25.get_scores(tokenized_query)

    top_indices = np.argsort(scores)[::-1][:5]
    top_5_tools = [tool_names[idx] for idx in top_indices]

    predictions.append(top_5_tools)

  r1, r5 = evaluate_recall(predictions, queries_data)
  return predictions, r1, r5

def evaluate_dense(model_name, tool_names, tool_descriptons, queries_data):
  model = SentenceTransformer(model_name)

  tool_embeddings = model.encode(tool_descriptions, convert_to_tensor=True, show_progress_bar=False)

  query_texts = [q['text'] for q in queries_data]
  query_embeddings = model.encode(query_texts, convert_to_tensor=True, show_progress_bar=False)

  search_results = util.semantic_search(query_embeddings, tool_embeddings, top_k=5)

  predictions = []
  for hits in search_results:
    top_5_tools = [tool_names[hit['corpus_id']] for hit in hits]
    predictions.append(top_5_tools)

  r1, r5 = evaluate_recall(predictions, queries_data)

  return predictions, r1, r5

if __name__ == "__main__":
  tool_names, tools_descriptions, queries_data = load_data()

  print("Evaluating BM25...")
  bm25_preds, bm25_r1, bm25_r5 = evaluate_bm25(tool_names, tool_descriptions, queries_data)

  print("Evaluating msmarco-MiniLM...")
  mini_preds, mini_r1, mini_r5 = evaluate_dense("sentence-transformers/msmarco-MiniLM-L-6-v3", tool_names, tool_descriptions, queries_data)

  print("Evaluating UAE-large-v1...")
  uae_preds, uae_r1, uae_r5 = evaluate_dense("WhereIsAI/UAE-Large-V1", tool_names, tool_descriptions, queries_data)


  print("\n--- Final Results for Report ---")
  print(f"{'Method':<25} | {'Recall@1':<10} | {'Recall@5':<10}")
  print("-" * 50)
  print(f"{'BM25':<25} | {bm25_r1:<10.4f} | {bm25_r5:<10.4f}")
  print(f"{'msmarco-MiniLM':<25} | {mini_r1:<10.4f} | {mini_r5:<10.4f}")
  print(f"{'UAE-large-v1':<25} | {uae_r1:<10.4f} | {uae_r5:<10.4f}")