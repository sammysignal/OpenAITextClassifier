import os
import pickle
import math
from openai import OpenAI

def load_embeddings():
    with open("mean_embeddings.pkl", "rb") as f:
        return pickle.load(f)


# Text classification using embeddings with nearest neighbors
class OpenAIClassifier:
    def __init__(
        self,
        example_data: list[dict],
        openai_key: str = os.getenv("OPENAI_API_KEY"),
        saved_mean_embeddings: list[list[float]] = [],
    ):
        self.openai = OpenAI(api_key=openai_key)
        if saved_mean_embeddings:
            self.mean_embeddings = saved_mean_embeddings
            self.label_count = len(saved_mean_embeddings)
        else:
            self._create_embeddings(example_data)

    # Create embeddings for the example data
    def _create_embeddings(self, example_data: list[dict]):
        mean_embeddings = []
        label_count = len(set(example["label"] for example in example_data))
        # Embeddings is an embedding for each example, for each label
        embeddings: list[list[list[float]]] = [[] for _ in range(label_count)]
        for example in example_data:
            text = example["text"]
            label = example["label"]
            embedding = self._embed(text)
            embeddings[label].append(embedding)
        # Calculate the mean embedding for each label
        for example_embeddings in embeddings:
            average_embedding = [0] * len(example_embeddings[0])
            for embedding in example_embeddings:
                for i in range(len(embedding)):
                    average_embedding[i] += embedding[i]
            for i in range(len(average_embedding)):
                average_embedding[i] /= len(example_embeddings)
            mean_embeddings.append(average_embedding)
        self.label_count = label_count
        self.mean_embeddings = mean_embeddings


    # Distance between two vector embeddings.
    def _d(self, a: list[float], b: list[float]) -> float:
        if len(a) != len(b):
            print(len(a), len(b))
            raise ValueError("Vectors must be of the same length")
        return sum([a[i] * b[i] for i in range(len(a))])

    # Embed a text using OpenAI's embedding model
    def _embed(self, text: str) -> list[float]:
        response = self.openai.embeddings.create(
            input=text, model="text-embedding-3-small"
        )
        return response.data[0].embedding

    # Classify a text using the mean embeddings of each label
    def classify(self, text: str) -> list[int]:
        embedding = self._embed(text)
        distances = [
            self._d(embedding, self.mean_embeddings[i]) for i in range(self.label_count)
        ]
        min = math.inf
        min_index = 0
        for i in range(len(distances)):
            if distances[i] < min:
                min = distances[i]
                min_index = i
        return min_index

    def save_embeddings(self):
        with open("mean_embeddings.pkl", "wb") as f:
            pickle.dump(self.mean_embeddings, f)

