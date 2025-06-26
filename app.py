from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import os
import time
import uuid
import threading
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
import librosa
import numpy as np
from openai import OpenAI

app = Flask(__name__)
app.secret_key = '8c2ac2a3-6284-4a81-9b9f-734293f17d5c'  # برای مدیریت session
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# یوزرنیم و پسورد ثابت
USERNAME = "admin"
PASSWORD = "1234"

tasks = {}

def run_heavy_task(filepath, task_id):
    result = heavy_function(filepath)
    tasks[task_id]["status"] = "done"
    tasks[task_id]["result"] = result

def heavy_function(file_name):

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "./whisper-large-v3-turbo"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=60,
        batch_size=16,
        torch_dtype=torch_dtype,
        device=device,
        generate_kwargs={"language": "fa"},
    )

    data, sr = librosa.load(file_name, sr=16000)

    chunk_duration = 60
    overlap_duration = 1
    chunk_samples = chunk_duration * sr
    overlap_samples = overlap_duration * sr
    step_samples = chunk_samples - overlap_samples

    chunks = []
    for start in range(0, len(data) - chunk_samples + 1, step_samples):
        end = start + chunk_samples
        chunk = data[start:end]
        chunks.append(chunk)
    outputs = ""

    api_key = "sk-local-key-dev"
    client = OpenAI(
        api_key=api_key,
        base_url="http://localhost:8090/v1",
    )
    
    SYSTEM_PROMPT = """You are an expert in Persian language editing. Your task is to revise and correct a Persian text transcribed from a 45-minute audio file of a Quranic session with a single speaker, ensuring the output is formal and literary. Follow these instructions precisely:

    1. **Preserve Original Sentences**: Do not change, rephrase, or replace any sentences or words with synonyms (e.g., do not replace "عزیزان" with "سروران" or "خیلی" with "بسیاری"). Only correct errors to maintain the original meaning.
    2. **Fix Transcription Errors**: Correct obvious mistakes, such as "غره دادن" instead of "قرار دادن," based on context to ensure semantic accuracy.
    3. **Convert Colloquial to Formal**:
    - Replace colloquial words with their formal equivalents: "ایشونو" → "ایشان را", "چیه" → "چیست", "کی" → "چه کسی", "چی" → "چه", "یه" → "یک", "چیزا" → "چیزها", "دیگه" → "دیگر", "آخه" → "آخر", "بریم" → "برویم".
    - Separate the "و" at the end of words (meaning "را") and write it as "را" (e.g., "ایشونو" → "ایشان را").
    4. **Handle Religious Phrases**:
    - Replace "علیه السّلام" with "(ع)", "سلام الله علیها" with "(س)", "صلی الله علیه و آله و سلم" with "(ص)", and "عجل الله تعالی فرجه" with "(عج)".
    5. **Verb Prefixes and Suffixes**:
    - Separate prefixes "می" and "نمی" from verbs with a non-breaking space (e.g., "می‌روم").
    - Remove the initial "ب" from verbs like "بکنیم" → "کنیم" or "بشویم" → "شویم" for a more literary tone.
    - Write possessive suffixes correctly with a non-breaking space (e.g., "مولامون" → "مولای‌مان").
    - Separate comparative suffixes like "تر", "تری", or "ترین" with a non-breaking space (e.g., "بهتر" → "به‌تر").
    6. **Correct Vowel Errors**: Fix colloquial vowel substitutions, such as "و" instead of "ا" (e.g., "ایشون" → "ایشان", "بچه‌هاشون" → "بچه‌های‌شان").
    7. **Preserve Specific Terms**: Retain words like "انصارالله", "انصاراللّهی", "دیّاران", or "قسوره" exactly as they appear, without using similar alternatives.
    8. **Apply Tashdid**: Ensure words with tashdid ("ّ") are written correctly (e.g., "اللّه").
    9. **Use Persian Numbers**: Replace Arabic or Latin numbers with Persian numerals (e.g., "١" or "1" → "۱").
    10. **Punctuation**: Use Persian punctuation marks (comma "،", period "۔", semicolon "؛", colon ":", question mark "؟", exclamation mark "!", etc.) correctly.
    11. **Output**: Return only the corrected and rewritten text without any additional explanations or metadata.

    Process the input text and produce a clean, formal Persian text file adhering to these rules.
    """

    output = ""

    for i, c in enumerate(chunks):
        result = pipe(c)
        stream = client.chat.completions.create(
        model="gemma-3-4b",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": result['text']},
            ],
            stream=True, 
            temperature=0.7,
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                output += chunk.choices[0].delta.content
                print(chunk.choices[0].delta.content, end="")

    return output

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        if request.form['username'] == USERNAME and request.form['password'] == PASSWORD:
            session['logged_in'] = True
            return redirect(url_for('upload_file'))
        else:
            return "ورود نامعتبر! برگرد و دوباره تلاش کن."
    return render_template('login.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            task_id = str(uuid.uuid4())
            tasks[task_id] = {"status": "processing", "result": None}

            # اجرای تابع در ترد جدا
            thread = threading.Thread(target=run_heavy_task, args=(filepath, task_id))
            thread.start()

            return redirect(url_for('loading_page', task_id=task_id))
    return render_template('upload.html')


@app.route('/process/<filename>')
def process_file(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    result = heavy_function(filepath)
    return render_template('result.html', result=result)


@app.route('/loading/<task_id>')
def loading_page(task_id):
    return render_template('loading.html', task_id=task_id)

@app.route('/api/status/<task_id>')
def task_status(task_id):
    task = tasks.get(task_id)
    if task:
        return jsonify(task)
    else:
        return jsonify({"status": "not_found"}), 404

@app.route('/result/<task_id>')
def show_result(task_id):
    task = tasks.get(task_id)
    if task and task["status"] == "done":
        return render_template('result.html', result=task["result"])
    else:
        return "نتیجه‌ای یافت نشد یا هنوز آماده نیست."


if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)
