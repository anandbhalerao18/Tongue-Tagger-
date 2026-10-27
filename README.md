# Tongue Tagger: language identification (character n-grams)

Web app that predicts the language of a text and marks language switches inside mixed sentences.

## Install
1. Install Python 3.9+ (https://python.org; tick "Add Python to PATH" on Windows).
2. Unzip this folder, open a terminal inside it.
3. (Optional) python -m venv venv, then activate it:
   Windows: venv\Scripts\activate    Mac/Linux: source venv/bin/activate
4. pip install -r requirements.txt     (only Flask is needed)
5. python app.py
6. Open http://127.0.0.1:5000

## How the NLP works
- Each word is padded with spaces and cut into character 1-, 2- and 3-grams.
- Training: n-gram frequency tables are built from data/<Language>.txt.
- Scoring: Naive Bayes log-probability with add-alpha smoothing, per word.
- Mixed text: a Viterbi pass over the words. Staying in one language is free,
  switching costs a penalty, so short ambiguous words follow their neighbours.

## Add a language
Drop a plain-text file named after the language (e.g. data/Tamil.txt) with a few
hundred words of real text, then restart. More text means better accuracy.
