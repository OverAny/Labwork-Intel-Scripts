from PIL import Image
import requests

from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

#Details#
url = "pandabike.png"
desc = "panda on a bicycle"
#(first item is correct, the rest are red herrings)#
lists = [["panda", "dog", "dragon", "cat", "lizard", "bug"], ["bicycle", "car", "truck", "bus", "plane", "popstick"]]
#-------#


for i in lists:
    image = Image.open(url)
    inputs = processor(text=i, images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
    print("----------------------")
    for x in range(len(i)):
        print(probs.detach().numpy()[0][x])
    
    result = True
    
    for z in range(len(i)):
        if (float(probs.detach().numpy()[0][0]) < float(probs.detach().numpy()[0][z])):
            result = False
    
    if (result):
        print("PASS!")
    else: 
        print("FAIL!")



