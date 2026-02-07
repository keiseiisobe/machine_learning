#!/bin/bash
# Virtual Environment Setup Script
# Works on macOS, Linux, and Windows (Git Bash/WSL)

set -e  # Exit on error

echo "Setting up virtual environment for machine learning repository..."

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment based on OS
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows (Git Bash)
    source venv/Scripts/activate
else
    # macOS/Linux
    source venv/bin/activate
fi

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo ""
echo "✓ Setup complete!"
echo ""
echo "To activate the virtual environment:"
echo "  - macOS/Linux:  source venv/bin/activate"
echo "  - Windows:      venv\\Scripts\\activate"
echo ""
echo "To deactivate:"
echo "  deactivate"
