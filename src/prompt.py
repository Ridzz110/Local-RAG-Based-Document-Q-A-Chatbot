from typing import List,Dict

def build_prompt(query: str, retrieved_chunks: List[Dict]) -> str :
    context_block = []
    for chunk in retrieved_chunks:
        source = chunk["metadata"].get("source", "unknown")
        page = chunk["metadata"].get("page", None)
        citation= f"{source}"
        if page is not None:
            citation += f" , page {page}"
        context_block.append(
            f"[source: {citation}]\n{chunk['text']}"
        )
    context = "\n\n".join(context_block)
    prompt = f'''
    You are a question-answering assistant.

    RULES (you MUST follow these):
    - You must answer using ONLY the information in the context.
    - Do NOT say you are an AI language model.
    - Do NOT give generic advice.
    - Do NOT mention access limitations.
    - If the answer is not present in the context, reply EXACTLY with:
    "I don't know based on the provided documents."

    Context:
    {context}

    Question:
    {query}

    Answer:
    '''.strip()

    return prompt
    
