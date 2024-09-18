from classification import OpenAIClassifier

# Example data for sentiment analysis
EXAMPLE_DATA = [
    # Negative
    {"text": "I hate chocolate!", "label": 0},
    {"text": "Today is a terrible day", "label": 0},
    {"text": "I am so sad about the news tonight :(", "label": 0},
    {"text": "I'd rather not put up with his attitude", "label": 0},
    {"text": "Our country has already endured many horrors and spikes of violence in the past, but what we experience right now are without a doubt the darkest hours of American society.", "label": 0},
    {"text": "I'm so disappointed in the way things are going", "label": 0},
    {"text": "Some say Prague is boring and not worth visiting.", "label": 0},
    {"text": "The wraps are not tasty and not worth their money.", "label": 0},

    # Positive
    {"text": "I love chocolate!", "label": 1},
    {"text": "Today is a beautiful day!", "label": 1},
    {"text": "I am so excited for the game tonight!", "label": 1},
    {"text": "It's so nice to see you", "label": 1},
    {"text": "I am so happy for you", "label": 1},
    {"text": "Prague is a beautiful city", "label": 1},
    {"text": "This coffee shop is amazing", "label": 1},
    {"text": "I am so grateful for all the blessings in my life", "label": 1},
    {"text": "I am so proud of you", "label": 1},
    {"text": "This restaurant has some really stupendous food.", "label": 1},

]

classifier = OpenAIClassifier(EXAMPLE_DATA)

print(classifier.classify("I love chocolate!"))
print(classifier.classify("Today is a terrible day!"))
print(classifier.classify("Where is the nearest coffee shop?"))
print(classifier.classify("I am not so sure about this one, boss"))


classifier.save_embeddings()

