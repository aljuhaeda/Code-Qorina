import streamlit as st
import pandas as pd
import numpy as np
import sklearn
import nltk
import sys
import importlib.metadata

def get_version(package_name):
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return "Package not found"

def print_versions():
    print("### Versions of Installed Packages:")
    print(f"Streamlit: {st.__version__}")
    print(f"Pandas: {pd.__version__}")
    print(f"NumPy: {np.__version__}")
    print(f"Scikit-learn: {sklearn.__version__}")
    print(f"NLTK: {nltk.__version__}")
    print(f"Sastrawi: {get_version('Sastrawi')}")
    print(f"Python: {sys.version}")

if __name__ == "__main__":
    print_versions()
