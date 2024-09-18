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

### test.py shows perfect accuracy for 5 test examples
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

## License

Copyright (c) 2024 Sameer Mehra, under the MIT license.

```
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```