# Text Classification using OpenAI Embeddings

This project is a simple text classification system that uses OpenAI's embedding model to classify
text into however many categories you want, using a simple k-nearest neighbors algorithm.

This works by creating embeddings for each example and then calculating the mean embedding for each
category. When you want to classify a new text, this process will create an embedding for that text
and then calculate the distance between the text embedding and each category mean embedding.
The category with the smallest distance is the category that the text is classified into.

## Usage

To use the classification system, you need to have an OpenAI API key. You can get one [here](https://platform.openai.com/account/api-keys).

See `test.py` for an example of how to use the classification system.

### test.py shows perfect accuracy for 5 test examples in each label

```
0
0
0
0
0
1
1
1
1
1
2
2
2
2
2
```
