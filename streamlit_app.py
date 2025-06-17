# Streamlit deployment entry point
# This file is used by Streamlit Cloud for deployment

import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(__file__))

# Import and run the main application
from app import main

if __name__ == "__main__":
    main() 