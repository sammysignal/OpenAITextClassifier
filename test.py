from classifier import OpenAIClassifier, load_embeddings

# Example data for sentiment analysis
EXAMPLE_DATA = [
    # Negative
    {"text": "I hate chocolate!", "label": 0},
    {"text": "Today is a terrible day", "label": 0},
    {"text": "I am so sad about the news tonight :(", "label": 0},
    {"text": "I'd rather not put up with his attitude", "label": 0},
    {"text": "I'm so disappointed in the way things are going", "label": 0},
    {"text": "Some say Prague is boring and not worth visiting.", "label": 0},
    {"text": "The wraps are not tasty and not worth their money.", "label": 0},
    {"text": "The project failed to meet even the most basic expectations.", "label": 0},
    {"text": "Every attempt to fix the issue only made things worse.", "label": 0},
    {"text": "The meeting was a complete waste of time and achieved nothing.", "label": 0},
    {"text": "It seems like there's no end in sight to this ongoing problem.", "label": 0},
    {"text": "The results were disappointing, falling far short of the goals.", "label": 0},
    {"text": "The team's lack of effort was painfully obvious.", "label": 0},
    {"text": "The product launch was a disaster from start to finish.", "label": 0},
    {"text": "Despite all the hard work, success feels unattainable.", "label": 0},
    {"text": "Customer feedback was overwhelmingly negative and disheartening.", "label": 0},
    {"text": "The situation has gone from bad to worse, with no clear solution in sight.", "label": 0},

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
    {"text": "You never cease to amaze me", "label": 1},
    {"text": "I am so glad I have you in my life", "label": 1},
    {"text": "I am so proud of you", "label": 1},
    {"text": "The project exceeded all expectations and was a huge success.", "label": 1},
    {"text": "Every effort made a positive difference, and the results were fantastic.", "label": 1},
    {"text": "The meeting was incredibly productive and sparked some great ideas.", "label": 1},
    {"text": "It feels like we're on the brink of something amazing.", "label": 1},
    {"text": "The results were outstanding, surpassing all our goals.", "label": 1},
    {"text": "The team's dedication and hard work were truly inspiring.", "label": 1},
    {"text": "The product launch was a smashing success, celebrated by everyone.", "label": 1},
    {"text": "With every challenge, we’re only getting closer to our goals.", "label": 1},
    {"text": "Customer feedback was overwhelmingly positive and uplifting.", "label": 1},
    {"text": "The situation keeps improving, and the future looks bright.", "label": 1},

    # Related to aliens
    {"text": "The spaceship hovered silently above the city, its alien technology beyond human comprehension.", "label": 2},
    {"text": "Strange symbols appeared in the desert, believed to be messages from an advanced alien civilization.", "label": 2},
    {"text": "Scientists were baffled by the signals coming from a distant galaxy, hinting at extraterrestrial intelligence.", "label": 2},
    {"text": "The alien creature, though fearsome in appearance, communicated with us using telepathy.", "label": 2},
    {"text": "In the remote mountains, people have reported seeing strange lights they believe to be alien spacecraft.", "label": 2},
    {"text": "The alien species had a language that was musical and complex, unlike anything we had ever heard.", "label": 2},
    {"text": "A mysterious artifact, seemingly of alien origin, was discovered buried deep beneath the ice.", "label": 2},
    {"text": "Many believe the government is hiding evidence of alien contact in top-secret facilities.", "label": 2},
    {"text": "The aliens came in peace, offering technologies that could transform our world.", "label": 2},
    {"text": "Ancient legends speak of gods descending from the stars, now thought to be early encounters with aliens.", "label": 2},
]

# # If previous embeddings exist, load them
# embeddings = load_embeddings()
# classifier = OpenAIClassifier(EXAMPLE_DATA, saved_mean_embeddings=embeddings)

# Otherwise, create new embeddings
classifier = OpenAIClassifier(EXAMPLE_DATA)
print("done creating embeddings")
# Save the embeddings
classifier.save_embeddings()

# Test Positive
print(classifier.classify("The celebration was filled with laughter and joy, bringing everyone together in happiness."))
print(classifier.classify("Each day seems to be getting better, filled with unexpected delights and smiles."))
print(classifier.classify("The team’s hard work paid off, and the atmosphere is electric with excitement."))
print(classifier.classify("Friends gathered around, sharing stories and creating memories that would last a lifetime."))
print(classifier.classify("The sunrise painted the sky in vibrant colors, filling the morning with a sense of peace and hope."))

# Test Negative
print(classifier.classify("The news left everyone in the room silent, a heavy cloud of sadness hanging over them."))
print(classifier.classify("Despite all the efforts, the project fell apart, leaving a sense of defeat in its wake."))
print(classifier.classify("The emptiness in the house was palpable after they left, each room echoing with the memories of happier times."))
print(classifier.classify("The storm destroyed everything they had worked so hard to build, leaving nothing but ruins."))
print(classifier.classify("Every attempt to make things right seemed to push them further into a cycle of despair."))

# Test Aliens
print(classifier.classify("The first contact with the alien beings revealed a language composed of light patterns and colors."))
print(classifier.classify("Astronomers were stunned when they discovered a new planet with signs of advanced alien architecture."))
print(classifier.classify("The alien ambassador extended a hand of friendship, its form both fascinating and otherworldly."))
print(classifier.classify("Stories of alien abductions have sparked fear and curiosity in small towns around the world."))
print(classifier.classify("The alien landscape was breathtakingly strange, with flora and fauna unlike anything on Earth."))

classifier.save_embeddings()

