from setuptools import setup, find_packages
from pathlib import Path

long_description = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

setup(
    name="plagiarism-detector",
    version="1.0.0",
    description="A practical NLP pipeline for catching copied and paraphrased text (TF-IDF + fuzzy + optional semantic)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your@email.com",
    url="https://github.com/your-username/plagiarism-detector",
    license="MIT",
    packages=find_packages(exclude=["tests*"]),
    python_requires=">=3.9",
    install_requires=[
        "scikit-learn>=1.3.0",
        "nltk>=3.8.0",
        "rapidfuzz>=3.0.0",
        "numpy>=1.24.0",
        "PyYAML>=6.0.0",
    ],
    extras_require={
        "semantic": ["spacy>=3.7.0"],
        "dev": ["pytest>=7.4.0", "pytest-cov>=4.1.0", "ruff"],
    },
    entry_points={
        "console_scripts": [
            "plagiarism-detector=cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    keywords="plagiarism nlp text-similarity tfidf cosine-similarity",
)
