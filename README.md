# Tongue Tagger

I built Tongue Tagger to identify the language of a piece of text and to show where a sentence switches from one language to another. It handles code-mixed text such as Hindi + English or Marathi + English, which is how many of us actually write messages.

## Features
- Detects 10 languages: English, Hindi, Marathi, Spanish, French, German, Italian, Portuguese, Dutch and Russian
- Splits mixed-language sentences and marks each word with its language
- Live results in the browser as you type, with confidence bars
- No machine learning libraries: the NLP is written from scratch in Python

## How it works
1. Every word is padded with spaces and cut into character 1-, 2- and 3-grams.
2. N-gram frequency profiles are built for each language from the text files in `data/`.
3. Each word is scored per language with Naive Bayes and smoothing.
4. A Viterbi pass over the words picks the best language for each one. Switching language costs a penalty, so short ambiguous words follow their neighbours.

## Requirements
- Python 3.9 or newer
- Flask (listed in `requirements.txt`)

## Run it
```
git clone https://github.com/anandbhalerao18/Tongue-Tagger-.git
cd Tongue-Tagger-
python -m pip install -r requirements.txt
python app.py
```
Then open http://127.0.0.1:5000 in your browser.

## Try these
- `Hello, मेरा नाम Rahul है and I live in Pune`
- `Je suis très heureux, but ich habe keine Zeit`
- `Mañana vamos a la escuela and then back home`

## Add a language
Put a plain-text file named after the language (for example `data/Tamil.txt`) with a few hundred words of real text, then restart the app. More text gives better accuracy.

## Project structure
```
app.py                Flask server and API (POST /api/detect)
langid.py             n-gram model, scoring and Viterbi segmentation
data/                 training text, one file per language
templates/index.html  web interface
```

## Limitations
- The training text is small, so very short or unusual sentences can be misjudged, and names like "Rahul" may be tagged with the wrong language.
- Romanized Hindi (Hindi in English letters) is not supported yet.

## Author
Anand Bhalerao
