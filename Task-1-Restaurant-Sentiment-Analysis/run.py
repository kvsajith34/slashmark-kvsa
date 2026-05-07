#!/usr/bin/env python3
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.app import app

if __name__ == "__main__":
    print("\n" + "="*60)
    print("???  RESTAURANT SENTIMENT ANALYSIS PROJECT")
    print("="*60)
    print("Starting web application...")
    print("Open your browser and go to: http://localhost:5000")
    print("="*60 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
