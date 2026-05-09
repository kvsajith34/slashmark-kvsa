# 🔍 Plagiarism Detector — NLP Pipeline

Hey there! I threw this together because I kept running into the same problem: students (and sometimes colleagues) submitting work that was a little *too* similar to stuff already out there. Manual checking is painful, so I built a solid little pipeline that does the heavy lifting for me.

It's not trying to be the next Turnitin — just a practical, no-nonsense tool that catches obvious copy-paste jobs, smart paraphrases, and near-duplicates using classic NLP tricks. Built with Python, scikit-learn, NLTK, and rapidfuzz. Works great locally or in CI.

---

## What it can do

I focused on the stuff that actually matters day-to-day:

- **TF-IDF + Cosine Similarity** — solid unigram/bigram matching with sublinear term frequency (catches vocabulary overlap really well)
- **Fuzzy matching** — rapidfuzz does the heavy lifting here. The `token_sort_ratio` scorer is surprisingly good at ignoring word order swaps and minor edits
- **Optional semantic similarity** — plug in spaCy’s `en_core_web_md` if you want to catch synonym-swapping paraphrases
- **Proper text cleanup** — Unicode normalization, URL stripping, punctuation handling, stopword removal, and lemmatization (falls back gracefully if NLTK data isn’t around)
- **Composite scoring** — blends the three methods with tunable weights so you’re not stuck with one rigid number
- **Reports that don’t suck** — clean standalone HTML + JSON you can actually read or parse
- **CLI + Python API** — use it from the terminal or drop it into your own scripts
- **CI friendly** — exits with code 1 when it finds something fishy (perfect for GitHub Actions)

---

## Quick start (the way I actually use it)

Clone it, install the bits you need:

```bash
git clone https://github.com/your-username/plagiarism-detector.git
cd plagiarism-detector
pip install -r requirements.txt
```

NLTK stuff downloads on first run (or it falls back to built-in lists if the network is being difficult — I made sure it still works in air-gapped environments).

### Try it on the sample docs right away

```bash
python cli.py --dir sample_documents/
```

You’ll get a nice summary table in the terminal plus HTML + JSON reports in the `reports/` folder. The samples include an original essay, a paraphrased version, a near-duplicate with tiny tweaks, and something completely unrelated — perfect for seeing how the scores behave.

### Checking your own stuff

```bash
# Whole folder
python cli.py --dir my_essays/ --threshold 0.70

# Just a few specific files
python cli.py --files essay1.txt essay2.txt essay3.txt

# Recursive + custom pattern (handy for submissions/)
python cli.py --dir submissions/ --recursive --pattern "*.txt" --semantic
```

---

## Configuration — tweak it until it feels right

Everything lives in `config.yaml`. I tried to make the defaults sensible, but you’ll probably want to adjust the threshold and weights for your use case.

```yaml
threshold: 0.75          # Anything above this gets flagged
ngram_range: [1, 2]      # Unigrams + bigrams usually works great
max_features: 10000
remove_stopwords: true
lemmatize: true
weights:
  tfidf: 0.5
  fuzzy: 0.3
  semantic: 0.2
fuzzy_method: token_sort_ratio
use_semantic: false
```

You can override anything from the command line too:

```bash
python cli.py --dir docs/ --threshold 0.65 --ngram-max 3 --semantic
```

---

## Python API (when you want to embed it)

```python
from plagiarism_detector import PlagiarismDetector, DetectorConfig, Reporter

config = DetectorConfig(threshold=0.70, ngram_range=(1, 3))
detector = PlagiarismDetector(config)

result = detector.check_directory("my_essays/")

# Or feed raw text directly
result = detector.check_texts({
    "Alice's draft": "...",
    "Bob's draft": "...",
})

reporter = Reporter(output_dir="reports/")
reporter.print_summary(result)
reporter.generate_html(result)
```

I use the `check_texts` path a lot when I’m pulling stuff from a database or Google Docs export.

---

## A note on semantic similarity

If you want the model to catch paraphrases that swap words around (e.g. “AI is changing industries” vs “Artificial intelligence is transforming sectors”), enable spaCy:

```bash
pip install spacy
python -m spacy download en_core_web_md
```

Then flip `use_semantic: true` in the config or pass `--semantic` on the CLI. It’s optional because the model is ~700MB and not everyone needs it.

---

## Running the tests (I actually do this)

```bash
pip install pytest pytest-cov
pytest tests/ -v --cov=plagiarism_detector
```

The test suite covers the preprocessor, vectorizer, similarity engine, full pipeline, and report generation. I added a few edge cases that bit me early on (empty docs, single-file runs, missing NLTK data, etc.).

---

## CI / GitHub Actions

The CLI returns exit code 1 whenever it flags anything. Super handy for blocking merges:

```yaml
- run: python cli.py --dir submissions/ --threshold 0.75 --no-html --quiet
```

---

## How the scores actually work (plain English version)

We calculate three numbers for every pair:

1. TF-IDF cosine (how much the important words overlap)
2. Fuzzy ratio (how similar the raw strings are after minor edits)
3. Optional spaCy semantic score (meaning similarity)

Then we blend them with weights that always add up to 1.0. A pair gets flagged if the final composite score meets or exceeds your threshold.

Rough guide I use:
- 0.90+ → basically the same document
- 0.75–0.89 → high similarity, worth a closer look
- 0.50–0.74 → moderate overlap — context matters
- < 0.50 → probably fine

---

## Roadmap / stuff I might add later

- Web UI (probably Streamlit, it’s quick)
- Pull text directly from URLs so you can compare against live web sources
- Store results in SQLite so you can track trends over time
- Better multilingual support (right now it’s very English-centric)
- Highlight exact matching sentences in the HTML report

If any of that sounds useful, open an issue or PR — happy to chat.

---

## License

MIT. Do whatever you want with it. Just don’t blame me if it misses something sneaky.

---

Built with ❤️ (and a healthy dose of frustration at bad academic writing) using scikit-learn, NLTK, rapidfuzz, and occasionally spaCy.

---

*Last updated: whenever I had a spare evening*
