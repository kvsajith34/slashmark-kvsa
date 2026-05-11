**Slashmark Internship Projects Collection**

**A curated portfolio of six practical AI and ML projects developed during the Slashmark internship program. Covering sentiment analysis, image classification, plagiarism detection, fraud detection, autonomous lane control, and AI-powered indoor navigation.**

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Tasks Overview](#tasks-overview)
  - [Task 1: Restaurant Sentiment Analysis](#task-1-restaurant-sentiment-analysis)
  - [Task 2: Cats vs Dogs Classifier](#task-2-cats-vs-dogs-classifier)
  - [Task 3: Plagiarism Checker](#task-3-plagiarism-checker)
  - [Task 4: Credit Card Fraud Detection](#task-4-credit-card-fraud-detection)
  - [Task 5: AI Self-Driving Cars](#task-5-ai-self-driving-cars)
  - [Task 6: AI-Powered Obstacle Detection](#task-6-ai-powered-obstacle-detection)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Screenshots / Demo](#screenshots--demo)
- [Future Improvements](#future-improvements)
- [Conclusion](#conclusion)
- [License](#license)

---

## Overview

This repository contains six independent yet thematically connected machine learning and AI projects completed as part of the Slashmark internship. Each task demonstrates a complete pipeline — from data handling and model development to deployment or simulation — showcasing practical skills in NLP, computer vision, anomaly detection, and autonomous systems.

The projects are organized as self-contained folders, each with its own documentation, code, notebooks, and requirements. The collection is ideal for internship submissions, portfolio presentations, and technical interviews.

---

## Repository Structure

```
slashmark-kvsa/
├── Task-1-Restaurant-Sentiment-Analysis/
│   ├── README.md
│   ├── requirements.txt
│   ├── run.py
│   ├── data/
│   ├── src/
│   ├── app/
│   ├── notebooks/
│   └── models/
├── Task-2-Cats-vs-Dogs-Classifier/
│   ├── README.md
│   ├── requirements.txt
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── notebook/
│   └── ...
├── Task-3-Plagiarism-checker/
│   ├── README.md
│   ├── requirements.txt
│   ├── cli.py
│   ├── config.yaml
│   ├── plagiarism_detector/
│   ├── sample_documents/
│   └── tests/
├── Task-4-Credit-card-Fraud-Detection/
│   ├── README.md
│   ├── requirements.txt
│   ├── train.py
│   ├── predict.py
│   ├── src/
│   ├── data/
│   └── outputs/
├── Task-5-Ai-Self-driving-Cars/
│   ├── README.md
│   ├── requirements.txt
│   ├── config.yaml
│   ├── scripts/run_pipeline.py
│   ├── src/
│   ├── data/
│   └── tests/
└── Task-6-Ai-Powered-Obstacle-Detection/
    └── aethernav/
        ├── README.md
        ├── requirements.txt
        ├── backend/
        ├── frontend/
        ├── configs/
        └── docs/
```

---

## Tasks Overview

### Task 1: Restaurant Sentiment Analysis

**Task Name:** Restaurant Sentiment Analysis  
**Objective:** Build a multi-class sentiment classifier (Positive / Negative / Neutral) for restaurant and product reviews.  
**Main Features:** Text preprocessing (tokenization, stop-word removal, lemmatization), TF-IDF with unigrams & bigrams, classical ML models, Flask web app with REST API and HTML interface.  
**Technologies Used:** Python, scikit-learn, NLTK, pandas, Flask, joblib, Jupyter Notebook.  
**Brief Working Explanation:** Raw text is preprocessed in `src/preprocessing.py`, vectorized with TF-IDF, and classified using Logistic Regression, SVM, or MultinomialNB. The best model is saved and served via a Flask application (`app/app.py` + `run.py`).  
**Output or Result:** Logistic Regression achieved **81.0% accuracy** and **80.9% F1-score**. Includes full Jupyter notebook workflow and a deployable web interface.

### Task 2: Cats vs Dogs Classifier

**Task Name:** Cats vs Dogs Classifier  
**Objective:** End-to-end CNN image classifier to distinguish cats from dogs.  
**Main Features:** Data preparation & augmentation, custom CNN with batch normalization and dropout, training with callbacks, evaluation, and single/batch prediction scripts.  
**Technologies Used:** Python, TensorFlow/Keras, Jupyter Notebook.  
**Brief Working Explanation:** Dataset is prepared and split, a 4-block CNN (32–256 filters) is trained on ~25,000 Kaggle images, and predictions are made on new images or videos.  
**Output or Result:** **95.51% validation accuracy** (best epoch 23), ~1.2 million parameters. Includes evaluation scripts, plots, and a complete Jupyter notebook.

### Task 3: Plagiarism Checker

**Task Name:** Plagiarism Checker  
**Objective:** Detect plagiarism and paraphrasing between text documents using a hybrid NLP approach.  
**Main Features:** TF-IDF + cosine similarity, rapidfuzz fuzzy matching, optional spaCy semantic similarity, configurable weighting, HTML/JSON reporting, CLI + Python API, CI/CD-ready exit codes.  
**Technologies Used:** Python, scikit-learn, NLTK, rapidfuzz, spaCy (optional), YAML config, pytest.  
**Brief Working Explanation:** Documents are preprocessed, vectorized, and scored using a composite similarity metric (TF-IDF + fuzzy + optional semantic). Results are output to terminal and rich reports.  
**Output or Result:** Terminal summary table + HTML/JSON reports in `reports/` folder. Exits with code 1 on plagiarism detection. Highly configurable via `config.yaml`.

### Task 4: Credit Card Fraud Detection

**Task Name:** Credit Card Fraud Detection  
**Objective:** Build a robust pipeline to detect fraudulent transactions in highly imbalanced data (<0.2% fraud).  
**Main Features:** SMOTE/undersampling strategies, feature engineering (log-amount, time features), multiple models (Logistic Regression, Random Forest, Gradient Boosting), cost-sensitive threshold tuning, cross-validation with resampling.  
**Technologies Used:** Python, scikit-learn, imbalanced-learn, pandas, numpy, matplotlib, seaborn, joblib.  
**Brief Working Explanation:** Loads Kaggle `creditcard.csv` (or synthetic data), applies resampling inside CV folds, trains models, tunes decision threshold based on business cost (FN ≈ $120, FP ≈ $2), and generates predictions with fraud scores.  
**Output or Result:** Saved `.joblib` models, ROC/PR curves, confusion matrices, threshold charts, and prediction CSV with `fraud_score` + `fraud_flag`. Full pipeline scripts and modular `src/` structure.

### Task 5: AI Self-Driving Cars

**Task Name:** AI Self-Driving Cars (Lane Detection & Control)  
**Objective:** Implement a modular autonomous driving stack focused on lane detection and PID-based lateral control.  
**Main Features:** Canny + Hough lane detection with perspective transform, polynomial/RANSAC fitting, PID controller with anti-windup, quantitative metrics (CTE, lane-keeping score, smoothness, FPS), YAML configuration.  
**Technologies Used:** Python, OpenCV, NumPy, YAML, pytest.  
**Brief Working Explanation:** Images/videos are processed through ROI masking, edge detection, and lane fitting to compute cross-track error (CTE), which is fed to a PID controller to generate steering commands. Annotated output videos and metrics are produced.  
**Output or Result:** Annotated videos with lane overlays and HUD, evaluation report with average |CTE|, lane-keeping percentage, steering smoothness, and FPS. Sample data and simulation mode included.

### Task 6: AI-Powered Obstacle Detection

**Task Name:** AI-Powered Obstacle Detection (AetherNav)  
**Objective:** Develop an AI-powered indoor obstacle avoidance system for robots and drones using perception, path planning, and real-time control.  
**Main Features:** YOLOv8 object detection + MiDaS depth estimation, A* path planning + potential fields, real-time dashboard (React + FastAPI WebSocket), Pygame simulator with sensor noise, YAML config, Docker-ready.  
**Technologies Used:** Python, PyTorch, Ultralytics (YOLOv8), FastAPI, React 18 + TypeScript + Tailwind, Pygame.  
**Brief Working Explanation:** Camera frames are analyzed for obstacles and depth; dynamic paths are planned and executed with safety monitoring. A full-stack web dashboard visualizes live feed, mini-map, telemetry, and AI decisions.  
**Output or Result:** Real-time dashboard, simulation environment with moving obstacles, production-grade backend/frontend architecture, and comprehensive documentation in `docs/`.

---

## Tech Stack

**Core Languages & Frameworks**  
- Python 3  
- Jupyter Notebook  

**Machine Learning & Deep Learning**  
- scikit-learn, TensorFlow/Keras, PyTorch, Ultralytics (YOLOv8)  

**Computer Vision & NLP**  
- OpenCV, NLTK, spaCy, rapidfuzz  

**Web & Deployment**  
- Flask, FastAPI (WebSocket), React 18 + TypeScript + Tailwind CSS  

**Data & Utilities**  
- pandas, NumPy, matplotlib, seaborn, joblib, imbalanced-learn, YAML, pytest  

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/kvsajith34/slashmark-kvsa.git
   cd slashmark-kvsa
   ```

2. (Recommended) Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate          # Windows: venv\Scripts\activate
   ```

3. Navigate to any task folder and install its dependencies:
   ```bash
   cd Task-X-...
   pip install -r requirements.txt
   ```

4. Additional setup notes (per task):
   - Task 1 & 3: May require `nltk.download()` on first run.
   - Task 4: Download `creditcard.csv` from Kaggle into `data/`.
   - Task 6 (AetherNav): May require Docker and additional model downloads (YOLOv8, MiDaS).

---

## Usage

Each task is self-contained. Refer to the individual `README.md` inside each folder for detailed instructions.

**Common patterns:**
- **Task 1:** `python run.py` → Open http://localhost:5000
- **Task 2:** `python train.py --epochs 25` then `python evaluate.py` or `python predict.py`
- **Task 3:** `python cli.py --dir sample_documents/`
- **Task 4:** `python train.py` then `python predict.py --input new_data.csv`
- **Task 5:** `python scripts/run_pipeline.py --mode video --input data/test_videos/solidWhiteRight.mp4`
- **Task 6 (AetherNav):** Follow `aethernav/README.md` for backend (`uvicorn`) + frontend (`npm run dev`) or simulator.

Jupyter notebooks are available in most tasks for interactive exploration.

---

## Dependencies

Each task maintains its own `requirements.txt`. Common packages across the repository include:

`scikit-learn`, `NLTK`, `Flask`, `TensorFlow`, `Keras`, `OpenCV`, `PyTorch`, `Ultralytics`, `FastAPI`, `pandas`, `NumPy`, `matplotlib`, `joblib`, `imbalanced-learn`, `rapidfuzz`, `PyYAML`, `pytest`

See individual task folders for exact versions and additional requirements (e.g., spaCy model, Kaggle dataset).

---
Recommended:
- Task 1: Flask web interface
- Task 2: Training curves + sample predictions
- Task 3: HTML report example
- Task 4: ROC/PR curves and confusion matrix
- Task 5: Annotated lane detection video frame
- Task 6: AetherNav dashboard screenshot
-->

---

## Future Improvements

- Create a unified root-level launcher and dashboard
- Add Docker Compose for all tasks
- Implement CI/CD with GitHub Actions across the repository
- Expand to transformer-based models (BERT, ViT)
- Add cloud deployment examples (Vercel, AWS, Hugging Face Spaces)
- Enhance test coverage and add performance benchmarks
- Create a consolidated portfolio website showcasing all tasks

---

## Conclusion

This repository presents a strong, diverse portfolio of real-world AI and machine learning projects. From classical NLP pipelines to modern computer vision and robotics systems, each task reflects careful engineering, clean code organization, and practical deployment considerations. It is well-suited for internship evaluations, technical interviews, and public portfolio use.

---

## License

Not mentioned in the repository root.  
Some individual tasks contain their own license files (e.g., MIT License in Task 5 and Task 6 subdirectories). Please check per-task folders for licensing details.

---
