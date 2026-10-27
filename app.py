import os
from flask import Flask, jsonify, render_template, request
from langid import LangID

app = Flask(__name__)
engine = LangID(os.path.join(os.path.dirname(__file__), "data"))


@app.route("/")
def index():
    return render_template("index.html", languages=engine.langs)


@app.route("/api/detect", methods=["POST"])
def detect():
    text = (request.get_json(silent=True) or {}).get("text", "")[:5000]
    return jsonify(engine.detect(text))


if __name__ == "__main__":
    print("Open http://127.0.0.1:5000 in your browser")
    app.run(debug=False, port=5000)
