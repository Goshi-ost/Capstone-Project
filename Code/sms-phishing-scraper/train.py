#!/usr/bin/env python3
"""
Quick training script for SMS Image Classifier
Run this from the sms-phishing-scraper directory
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from Tools.smsImageClassifier import train

if __name__ == "__main__":
    resources = Path(__file__).resolve().parents[2] / "Resources"
    images_path = resources / "smsMessages"
    non_sms_path = resources / "NonSMSMessages"
    
    print("SMS Image Classifier Training")
    print("=" * 50)
    print(f"Images directory: {images_path}")
    print(f"Non-SMS directory: {non_sms_path}")
    print()
    
    try:
        train(images_path, epochs=15, batch_size=32, lr=1e-3, non_sms_dir=non_sms_path)
        print("\n Training complete!")
    except Exception as e:
        print(f"\n Error: {e}")
        sys.exit(1)
