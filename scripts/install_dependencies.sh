#!/bin/bash
# Install dependencies for the Scientific Gaze Zone Evaluation GUI

echo "Installing dependencies for Scientific Gaze Zone Evaluation GUI..."

# Check if we're in a virtual environment
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "WARNING: You are not in a virtual environment."
    echo "It's recommended to use a virtual environment for Python projects."
    echo "Continue anyway? (y/n)"
    read -r response
    if [[ "$response" != "y" ]]; then
        echo "Installation canceled."
        exit 1
    fi
fi

# Install required Python packages
echo "Installing Python packages..."
pip install numpy pandas matplotlib seaborn scipy scikit-learn PyQt6 openpyxl

echo "Checking installations..."
python -c "import numpy; print('NumPy:', numpy.__version__)"
python -c "import pandas; print('pandas:', pandas.__version__)"
python -c "import matplotlib; print('matplotlib:', matplotlib.__version__)"
python -c "import seaborn; print('seaborn:', seaborn.__version__)"
python -c "import scipy; print('scipy:', scipy.__version__)"
python -c "import sklearn; print('scikit-learn:', sklearn.__version__)"
python -c "import PyQt6; print('PyQt6:', PyQt6.__version__)"
python -c "import openpyxl; print('openpyxl:', openpyxl.__version__)"

echo "Installation complete!"
echo "You can now run the GUI with: python scripts/evaulation_gui.py --gui" 