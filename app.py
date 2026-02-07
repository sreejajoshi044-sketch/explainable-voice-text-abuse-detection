import os
import numpy as np
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import torch
from transformers import pipeline
import shap
from pydub import AudioSegment
import whisper

# ----------------------------------------
# INITIAL SETUP
# ----------------------------------------
print("🎧 Loading AI models...")
app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Disable gradients for better performance
torch.set_grad_enabled(False)

# ----------------------------------------
# MODEL LOADING
# ----------------------------------------

# ✅ Lightest Whisper model for English
whisper_model = whisper.load_model("tiny.en")

# ✅ Balanced toxicity model (less false positives)
toxicity_pipeline = pipeline(
    "text-classification",
    model="SkolkovoInstitute/roberta_toxicity_classifier"
)

# SHAP explainer for word-level visualization
shap_explainer = shap.Explainer(toxicity_pipeline)


# ----------------------------------------
# HELPER FUNCTIONS
# ----------------------------------------

def convert_to_wav(input_path):
    """Convert any audio file to mono 16kHz WAV."""
    output_path = os.path.splitext(input_path)[0] + ".wav"
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio.export(output_path, format="wav")
    return output_path


def get_explanation_html(shap_values):
    """Generate color-coded SHAP visualization for words."""
    words = shap_values.data[0]
    values = shap_values.values[0]
    max_val = np.abs(values).max()
    html_parts = []

    for word, val in zip(words, values):
        shap_val = val[0]
        color = "255,0,75" if shap_val > 0 else "0,139,251"
        opacity = abs(shap_val) / max_val if max_val > 0 else 0
        html_parts.append(
            f'<span style="background-color:rgba({color},{opacity:.2f}); '
            f'padding:2px 4px; border-radius:3px;">{word}</span>'
        )
    return " ".join(html_parts)


def analyze_text(text):
    """Analyze text toxicity with calibration."""
    try:
        result = toxicity_pipeline([text])[0]
        score = result.get("score", 0.0)
        label = result.get("label", "unknown")

        # Calibration: reduce bias for short/neutral text
        adjusted_score = score
        if label.lower() in ["toxic", "toxic_comment", "toxic-bert", "label_1"]:
            adjusted_score = max(0.0, score - 0.5)

        # Categorize
        if adjusted_score > 0.7:
            category = "Abusive"
        elif adjusted_score > 0.3:
            category = "Inappropriate"
        else:
            category = "Civil"

        # SHAP explanation
        try:
            shap_values = shap_explainer([text])
            explanation_html = get_explanation_html(shap_values)
        except Exception as e:
            print("⚠️ SHAP failed:", e)
            explanation_html = "<i>Explanation unavailable</i>"

        return {
            "toxicity_score": f"{adjusted_score:.4f}",
            "category": category,
            "explanation_html": explanation_html,
        }

    except Exception as e:
        print("❌ Text analysis failed:", e)
        return {"error": str(e)}


# ----------------------------------------
# ROUTES
# ----------------------------------------

@app.route('/')
def home():
    return render_template('landing.html')


@app.route('/analyzer-page')
def analyzer_page():
    return render_template('analyzer.html')


@app.route('/text-analyzer')
def text_analyzer_page():
    return render_template('text_analyzer.html')


# ---------- AUDIO ANALYZER ----------
@app.route('/analyze-audio', methods=['POST'])
def analyze_audio_api():
    print("🎧 Received audio for analysis")

    if 'audio_file' not in request.files:
        return jsonify({'error': 'No audio file uploaded'}), 400

    file = request.files['audio_file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    print(f"✅ File saved: {filepath}")

    wav_path = None
    try:
        wav_path = convert_to_wav(filepath)
        print("🎙 Transcribing...")
        transcription_result = whisper_model.transcribe(wav_path, fp16=False)
        text = transcription_result.get("text", "").strip()
        print("📝 Transcribed text:", text)

        if not text:
            return jsonify({'error': 'No speech detected'}), 400

        result = analyze_text(text)
        result["transcribed_text"] = text
        return jsonify(result)

    except Exception as e:
        print("❌ Audio analysis failed:", e)
        return jsonify({'error': str(e)}), 500

    finally:
        for path in [filepath, wav_path]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                    print(f"🧹 Deleted: {path}")
                except PermissionError:
                    print(f"⚠️ Could not delete {path} (in use)")


# ---------- TEXT ANALYZER ----------
@app.route('/analyze-text', methods=['POST'])
def analyze_text_api():
    print("🧠 Received text analysis request")

    if request.is_json:
        data = request.get_json()
        text = data.get("text", "").strip()
    else:
        text = request.form.get("text", "").strip()

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    result = analyze_text(text)
    print("✅ Text analysis result:", result)
    return jsonify(result)


# ----------------------------------------
# RUN APP
# ----------------------------------------
if __name__ == '__main__':
    app.run(debug=True)
