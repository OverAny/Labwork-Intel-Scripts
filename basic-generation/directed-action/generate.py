import csv
from random import choice

verbs = ["Hitting", "Kicking", "Poking", "Biting", "Pointing at", "Looking at", "Sitting on"]
animals = ["Dog", "Cat", "Rat", "Mouse", "Horse", "Pig", "Cow", "Goat", "Lamb", "Deer", "Moose", "Elk", "Raccoon", "Squirrel", "Chipmunk", "Skunk", "Fox", "Wolf", "Owl", "Hawk", "Falcon", "Eagle", "Vulture", "Condor", "Raven", "Crow", "Pigeon", "Dove", "Peacock", "Parrot", "Cockatoo", "Penguin", "Seagull", "Albatross", "Flamingo", "Swan", "Duck", "Goose", "Chicken", "Turkey", "Owl", "Beaver", "Rabbit", "Hare", "Giraffe", "Zebra", "Kangaroo", "Wallaby", "Koala", "Wombat", "Badger", "Hedgehog", "Otter", "Weasel", "Mink", "Ferret", "Seal", "Sea Lion", "Walrus", "Tiger", "Lion"]
objects = ["a book",    "a flower",    "a guitar",    "a hat",    "a pen",     "a spoon",    "a watch",    "a ball of yarn",     "a bag",    "a chair",    "a doll",   "a water bottle"]

sentences = []
lists = []
for i in range(100):
    # Choose a random animal, verb, and object
    animal = choice(animals)
    verb = choice(verbs).lower()
    obj = choice(objects).lower()
    
    # Create a sentence using the random words
    sentence = f'{animal} {verb} {obj}.'
    
    # Add the sentence and the individual words to the list of sentences
    sentences.append((sentence, animal.lower(), verb, obj))
    lists.append

# Write the sentences to a CSV file
with open('directed_action.csv', 'w', newline='') as csvfile:
  # Create a CSV writer
  writer = csv.writer(csvfile)
  
  # Write the header row
  writer.writerow(["sentence", "first", "action", "second"])
  
  # Write each sentence and its words to the CSV file
  for sentence in sentences:
    # Split the sentence into words
    # Write the sentence, first word, action, and second word to the CSV file
    writer.writerow([sentence[0], sentence[1], sentence[2], sentence[3]])
