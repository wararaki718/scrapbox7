from google import genai
from tqdm import tqdm


class GeminiClassifier:
    def __init__(self, model: str = "gemini-2.5-flash") -> None:
        self._client = genai.Client()
        self._model = model

    def estimate(self, prompt: str, testdata: str) -> list[int]:
        y_preds = []
        for text in tqdm(testdata["text"]):
            content = prompt.replace("[DOCUMENT]", text)
            response = self._client.models.generate_content(
                model=self._model,
                contents=content,
            )
            y_pred = int(response.text.strip())
            y_preds.append(y_pred)
        return y_preds
    