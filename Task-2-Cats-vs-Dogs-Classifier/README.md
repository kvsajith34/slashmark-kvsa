# 🐱 vs 🐶 — Cats & Dogs CNN Classifier

**Task 2 | Deep Learning Workflow — End to End**  
**Final Validation Accuracy: 95.51%**

A clean, complete end-to-end Convolutional Neural Network project to classify cats vs dogs using TensorFlow/Keras.

---

📊 Project Summary

| Metric                    | Value          |
|---------------------------|----------------|
| **Validation Accuracy**   | **95.51%**     |
| **Dataset**               | Full Kaggle (25,000 images) |
| **Best Epoch**            | 23             |
| **Total Parameters**      | ~1.2 Million   |
| **Training Time**         | ~45 minutes    |

---

🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/mark-kvs/cats-vs-dogs.git
cd cats-vs-dogs

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Prepare dataset
python setup_data.py                    # Sample dataset
# OR
python setup_data.py --source kaggle --kaggle-dir path/to/kaggle/train

# 5. Train the model
python train.py --epochs 25

# 6. Evaluate performance
python evaluate.py

 📁 Project Structure

cats_vs_dogs/
├── model.py                 # CNN architecture
├── train.py                 # Training with callbacks
├── evaluate.py              # Full evaluation suite
├── predict.py               # Single/batch prediction
├── setup_data.py            # Dataset preparation
├── requirements.txt
├── notebook/
│   └── cats_vs_dogs.ipynb   # Complete Jupyter notebook
├── README.md
└── .gitignore

🧠 Model Architecture

4 Convolutional Blocks (32 → 64 → 128 → 256 filters)
Batch Normalization + MaxPooling after each block
Global Average Pooling
Dense(256) + Dropout(0.5)
Sigmoid output

Regularization: L2, Dropout, BatchNorm, Data Augmentation

📈 Key Results

Validation Accuracy: 95.51%
Very balanced performance on cats and dogs
Strong ROC curve and low misclassification rate (only 224 errors out of 4,991 images)
