from dotenv import load_dotenv
load_dotenv()

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import Faithfulness, ContextRecall, FactualCorrectness, SemanticSimilarity
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from retrieve import retrieve_relevant_chunks
from prompt import build_prompt
from llm import generate_answer
import os
import warnings
warnings.filterwarnings("ignore")

# ── Configure RAGAS ───────────────────────────────────────────────────────────
evaluator_llm = LangchainLLMWrapper(
    ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY")
    )
)

evaluator_embeddings = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
)

metrics = [
    Faithfulness(llm=evaluator_llm),
    ContextRecall(llm=evaluator_llm),
    FactualCorrectness(llm=evaluator_llm),
    SemanticSimilarity(embeddings=evaluator_embeddings),
]

# ── Test dataset ──────────────────────────────────────────────────────────────
test_cases = [
    {
        "question": "What is FOL?",
        "ground_truth": "FOL stands for First Order Logic. It models the world in terms of objects, properties, and relations and allows quantification over variables."
    },
    {
        "question": "What is higher order logic?",
        "ground_truth": "Higher order logic allows quantification over relations and functions, unlike FOL which only quantifies over objects."
    },
    {
        "question": "What is the domain in FOL semantics?",
        "ground_truth": "The domain M is the set of all objects in the world of interest used in FOL semantics."
    },
]

# ── Run RAG pipeline ──────────────────────────────────────────────────────────
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
questions, answers, contexts, ground_truths = [], [], [], []

print("Running RAG pipeline on test cases...\n")

for tc in test_cases:
    question = tc["question"]
    chunks = retrieve_relevant_chunks(question, embedding_model, top_k=3)
    prompt = build_prompt(question, chunks)
    answer = generate_answer(prompt)

    questions.append(question)
    answers.append(answer)
    contexts.append([c["text"] for c in chunks])
    ground_truths.append(tc["ground_truth"])

    print(f"Q: {question}")
    print(f"A: {answer[:120]}...")
    print()

# ── Evaluate ──────────────────────────────────────────────────────────────────
eval_dataset = Dataset.from_dict({
    "user_input":         questions,
    "response":           answers,
    "retrieved_contexts": contexts,
    "reference":          ground_truths,
})

print("Running RAGAS evaluation...\n")

results = evaluate(
    dataset=eval_dataset,
    metrics=metrics,
)

print("\n===== RAGAS Evaluation Results =====")
df = results.to_pandas()
print(df[["user_input", "faithfulness", "context_recall", "factual_correctness(mode=f1)", "semantic_similarity"]])
df.to_csv("eval_results.csv", index=False)
print("\nResults saved to eval_results.csv")