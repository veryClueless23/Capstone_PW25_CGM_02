from flask import Flask, request, jsonify, render_template_string
import os

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200 MB max


# --- Define folders ---
CSV_DIR = "csv_logs"
ZIP_DIR = "foregrounds_new_zips"

# --- Create them if they don’t exist ---
os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(ZIP_DIR, exist_ok=True)

# --- Simple HTML dashboard ---
TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Raspberry Pi Upload Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; background: #f8f9fa; padding: 20px; }
        h1 { color: #333; }
        h2 { margin-top: 40px; }
        .success { color: green; }
        .warning { color: orange; }
        .file-list { margin-left: 20px; }
    </style>
</head>
<body>
    <h1>📦 Raspberry Pi Upload Dashboard</h1>

    <h2>CSV Logs</h2>
    {% if csv_files %}
        <p class="success">{{ csv_files|length }} CSV file(s) received:</p>
        <ul class="file-list">
            {% for f in csv_files %}
                <li>{{ f }}</li>
            {% endfor %}
        </ul>
    {% else %}
        <p class="warning">No CSV logs received yet.</p>
    {% endif %}

    <h2>🗜 Foregrounds Archives</h2>
    {% if zip_files %}
        <p class="success">{{ zip_files|length }} ZIP file(s) received:</p>
        <ul class="file-list">
            {% for f in zip_files %}
                <li>{{ f }}</li>
            {% endfor %}
        </ul>
    {% else %}
        <p class="warning">No ZIP files received yet.</p>
    {% endif %}
</body>
</html>
"""

# --- File upload endpoint ---
@app.route("/upload", methods=["POST"])
def upload_file():
    print("FILES RECEIVED:", request.files)

    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    filename = file.filename

    if not filename:
        return jsonify({"error": "No filename provided"}), 400

    if filename.endswith(".csv"):
        save_path = os.path.join(CSV_DIR, filename)
    elif filename.endswith(".zip"):
        save_path = os.path.join(ZIP_DIR, filename)
    else:
        return jsonify({"error": "Unsupported file type"}), 400

    file.save(save_path)
    print(f"Received {filename} → {save_path}")
    return jsonify({"message": f"File {filename} uploaded successfully!"})
# --- Dashboard route ---
@app.route("/")
def dashboard():
    csv_files = sorted(os.listdir(CSV_DIR))
    zip_files = sorted(os.listdir(ZIP_DIR))
    return render_template_string(TEMPLATE, csv_files=csv_files, zip_files=zip_files)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)