#!/usr/bin/env python3
"""
Quick training script for SMS Image Classifier
Run this from the sms-phishing-scraper directory
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from Tools.smsImageClassifier import train

if __name__ == "__main__":
    # Train on images_out folder
    images_path = "../../Resources/images_out"
    
    print("SMS Image Classifier Training")
    print("=" * 50)
    print(f"Images directory: {images_path}")
    print()
    
    try:
        train(images_path, epochs=15, batch_size=32, lr=1e-3)
        print("\n Training complete!")
    except Exception as e:
        print(f"\n Error: {e}")
        sys.exit(1)
