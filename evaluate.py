"""
RAG evaluation using RAGAS.
Run: python3 evaluate.py

Scores:
  faithfulness     — is the answer grounded in retrieved chunks? (not hallucinated)
  answer_relevancy — does the answer actually address the question?
  context_recall   — did retrieval find chunks that contain the answer?
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from dotenv import load_dotenv
from ragas import evaluate, EvaluationDataset
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from agent import ask
import os

load_dotenv()

_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY")))
_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY")))

faithfulness.llm = _llm
answer_relevancy.llm = _llm
answer_relevancy.embeddings = _embeddings
context_recall.llm = _llm

metrics = [faithfulness, answer_relevancy, context_recall]

# --- GOLDEN DATASET ---
# questions we know the answers to from our docs
# ground_truth is what a correct answer should contain
golden_dataset = [
    {
        "question": "What is the transformer architecture?",
        "ground_truth": "The transformer is a family of neural network architectures based on the multi-head attention mechanism, introduced in the 2017 paper Attention Is All You Need."
    },
    {
        "question": "What is a large language model?",
        "ground_truth": "A large language model is a type of AI model trained on large amounts of text data that can generate and understand natural language."
    },
    {
        "question": "What is cognitive psychology?",
        "ground_truth": "Cognitive psychology is the scientific study of mental processes such as attention, memory, perception, language, and problem solving."
    },
    {
        "question": "What is the attention mechanism in transformers?",
        "ground_truth": "The attention mechanism allows the model to focus on different parts of the input by computing query, key and value matrices to determine relevance between tokens."
    },
    {
        "question": "What is working memory?",
        "ground_truth": "Working memory is a cognitive system that temporarily holds and manipulates information, often described using the Baddeley and Hitch model."
    },
    {
        "question": "What is BERT?",
        "ground_truth": "BERT is an encoder-only transformer model developed by Google, trained using masked language modeling for representation learning."
    },
    {
        "question": "What is reinforcement learning from human feedback?",
        "ground_truth": "RLHF is a technique used to fine-tune language models using human preferences as a reward signal to align model outputs with human values."
    },
    {
        "question": "What is the capital of France?",
        "ground_truth": "I don't know based on the provided context."
    },
]

# --- RUN PIPELINE AND COLLECT RESULTS ---
print("Running RAG pipeline on golden dataset...\n")

samples = []
for item in golden_dataset:
    result = ask(query=item["question"], history=[])
    answer = result["answer"]
    contexts = [c.page_content for c in result.get("raw_chunks", [])]

    print(f"Q: {item['question']}")
    print(f"A: {answer[:120]}...")
    print(f"Contexts retrieved: {len(contexts)}\n")

    samples.append({
        "user_input": item["question"],
        "response": answer,
        "retrieved_contexts": contexts if contexts else [answer],
        "reference": item["ground_truth"],
    })

# --- EVALUATE ---
print("Evaluating with RAGAS...\n")
dataset = EvaluationDataset.from_list(samples)
results = evaluate(
    dataset=dataset,
    metrics=metrics,
)

print("\n── RAGAS Scores ──────────────────")
print(results)
