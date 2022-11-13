import os

import openai
from flask import Flask, redirect, render_template, request, url_for

app = Flask(__name__)
openai.api_key = "-----"


@app.route("/", methods=("GET", "POST"))
def index():
    if request.method == "POST":
        response = openai.Completion.create(
            model="text-davinci-002",
            prompt=generate_prompt( ),
            temperature=0.6,
            max_tokens=256,


        )
        return redirect(url_for("index", result=response.choices[0].text))

    result = request.args.get("result")
    return render_template("index.html", result=result)


def generate_prompt():
    return """Create 5 groups of three exam questions each, where first question requires the student to draw an image of a chimera creature, then the other two ask if all the parts are present. For example: - Draw a lion with the head of an ox. - Is the body of the chimera a lion body? - Is the head of the chimera an ox head?
    Questions: 
    """
    # .format(
    #     animal.capitalize()
    # )