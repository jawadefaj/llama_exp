#!/bin/bash

# Exit on error
set -e

echo "🔧 Installing transformers..."
pip install transformers

echo "🔧 Installing Hugging Face CLI with CLI extras..."
pip install -U "huggingface_hub[cli]"

echo "🔧 Installing TensorBoard..."
pip install tensorboard

echo ""
echo "✅ All packages installed."
echo "👉 Now authenticating with Hugging Face CLI..."

# Prompt for Hugging Face token
# read -p "🔐 Enter your Hugging Face token: " HF_TOKEN

# Login using the CLI
huggingface-cli login --token "$HF_TOKEN"

echo "🎉 Setup complete!"
