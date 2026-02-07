# Core AI libraries
torch==2.8.0+cpu
torchvision==0.23.0+cpu
torchaudio==2.8.0+cpu
numpy==1.25.0

# Whisper for transcription
git+https://github.com/openai/whisper.git@main

# Transformers for toxicity pipeline
transformers==4.44.2
huggingface-hub==0.18.0

# Explanation library
shap==0.42.1

# Flask web framework
Flask==2.3.3
Werkzeug==2.3.8

# Optional: for handling PDFs and DOCX files
PyPDF2==3.0.1
python-docx==0.8.12

# Optional: sanitize HTML
bleach==6.1.0
