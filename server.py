"""
Flask web server for the Study Assistant.
Serves the HTML chat frontend and exposes /ask and /quiz endpoints.
"""

from flask import Flask, request, jsonify, render_template
from rag_engine import StudyAssistant

app = Flask(__name__)

print("Starting up Study Assistant (this loads your notes and the model)...")
assistant = StudyAssistant()
print("Ready!")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(force=True)
    question = (data.get("message") or "").strip()
    if not question:
        return jsonify({"error": "No message provided"}), 400
    answer = assistant.ask(question)
    return jsonify({"answer": answer})


@app.route("/quiz", methods=["POST"])
def quiz():
    data = request.get_json(force=True)
    topic = (data.get("topic") or "the material").strip()
    quiz_text = assistant.quiz(topic)
    return jsonify({"quiz": quiz_text})


@app.route("/reset", methods=["POST"])
def reset():
    assistant.reset_memory()
    return jsonify({"status": "memory cleared"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
