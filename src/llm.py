import ollama

def generate_answer(prompt: str, model: str = 'phi', temperature: float = 0.0) -> str:
    response = ollama.chat(
        model=model,
        messages = [{"role": "user", "content": prompt}],
        options={
            "temperature": temperature
        }
    )
    return response.message.content.strip()