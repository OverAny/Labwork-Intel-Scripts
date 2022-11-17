from PIL import Image
import requests
import openai
import os 
import openai

from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

openai.api_key = "--------"

#Details#
url = "pandabike.png"
desc = "panda on a bicycle"
#(first item is correct, the rest are red herrings)#
# lists = [["panda", "dog", "dragon", "cat", "lizard", "bug"], ["bicycle", "car", "truck", "bus", "plane", "popstick"]]
lists = []
#-------#


response = openai.Completion.create(
    model="text-davinci-002",
    prompt="List the objects in the following sentence (create panda on a bicycle) seperated by commas:",
    temperature=0,
    max_tokens=256,
)


print(response.choices[0].text.strip())

objects = response.choices[0].text.strip().split(", ")
print(objects)

for i in objects:
    response = openai.Completion.create(
    model="text-davinci-002",
    prompt="list 4 one word objects that are different from the word " + i + " seperated by commas",
    temperature=0,
    max_tokens=256,
    )
    # print(response.choices[0].text.strip())
    objectsNew = response.choices[0].text.strip().split(", ")
    # print(objectsNew)
    objectsNew.insert(0, i)
    lists.append(objectsNew)

print(lists)

for i in lists:
    image = Image.open(url)
    inputs = processor(text=i, images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
    print("----------------------")    
  
    result = True
    
    for z in range(len(i)):
        print(i[z] + ": " + str(probs.detach().numpy()[0][z]))
        if (float(probs.detach().numpy()[0][0]) < float(probs.detach().numpy()[0][z])):
            result = False
    
    if (result):
        print(" ## " + i[0] + ": " + "PASS!")
    else: 
        print(" ## " + i[0] + ": " + "FAIL!")

print("----------------------")    


