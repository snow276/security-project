#!/bin/bash
# Setup script for AlertBERT + LLM-assisted SOC Alert Analysis
# Course project submission

set -e

echo "============================================"
echo "AlertBERT + LLM Alert Analysis Setup"
echo "============================================"

# 1. Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

# 2. Install Python dependencies
echo ""
echo "[1/5] Installing Python dependencies..."
pip install -r requirements.txt

# 3. Install AlertBERT package
echo ""
echo "[2/5] Installing AlertBERT package..."
cd AlertBERT
pip install -e .
cd ..

# 4. Apply AlertBERT patch (our modifications)
echo ""
echo "[3/5] Applying AlertBERT modifications..."
if [ -f patches/alertbert_models.patch ]; then
    cd AlertBERT
    git apply ../patches/alertbert_models.patch || echo "Patch may already be applied or AlertBERT is already modified"
    cd ..
else
    echo "Warning: Patch file not found. AlertBERT may be unmodified."
fi

# 5. Create API key placeholder
echo ""
echo "[4/5] Setting up API key..."
if [ ! -f deepseek-apikey.txt ]; then
    echo "YOUR_DEEPSEEK_API_KEY_HERE" > deepseek-apikey.txt
    echo "  Created deepseek-apikey.txt -- PLEASE EDIT THIS FILE WITH YOUR ACTUAL API KEY"
else
    echo "  deepseek-apikey.txt already exists"
fi

# 6. Check for AlertBERT model checkpoint
echo ""
echo "[5/5] Checking model checkpoint..."
CHECKPOINT="AlertBERT/saved_models/mlm_1l_4h_16d_original_default_params_60k.pt"
if [ ! -f "$CHECKPOINT" ]; then
    echo "  Warning: Model checkpoint not found at $CHECKPOINT"
    echo "  Please download the pretrained AlertBERT model and place it in AlertBERT/saved_models/"
    echo "  See AlertBERT/README.md for download instructions."
else
    echo "  Model checkpoint found."
fi

echo ""
echo "============================================"
echo "Setup complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Edit deepseek-apikey.txt with your DeepSeek API key"
echo "  2. Download AIT-ADS-A dataset (see README.md)"
echo "  3. Run baseline: python scripts/run_baseline.py"
echo "  4. Run LLM analysis: python scripts/run_llm_analysis.py"
echo ""
