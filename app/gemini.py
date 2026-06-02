import time
from google import genai
from app.config import GEMINI_API_KEY, GEMINI_MODEL

_client = genai.Client(api_key=GEMINI_API_KEY)


def generate(prompt: str) -> str:
    delays = [2, 4, 8]
    last_error = None

    for attempt, delay in enumerate(delays):
        try:
            response = _client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            )
            return response.text
        except Exception as e:
            last_error = e
            if attempt < len(delays) - 1:
                time.sleep(delay)

    raise RuntimeError(f"Gemini generation failed after 3 attempts: {last_error}")
