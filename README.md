# AI Learning & Study Assistant (Local, with Ollama)

A simple study chatbot that answers questions from your notes (RAG),
remembers the conversation (Memory), and generates quizzes.

## 1. Install Ollama

Download from https://ollama.com and install it.

Then pull a model (do this once):

```
ollama pull llama3
```

Keep Ollama running in the background (it usually starts automatically after install).

## 2. Set up Python

Create a virtual environment (recommended) and install dependencies:

```
python3 -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Add your notes

Replace `data/sample_notes.txt` with your own course material,
or add more `.txt` files and update `DATA_FILE` in `app.py`.
(PDF support can be added later with `PyPDFLoader`.)

## 4. Run it

**Option A: Command line version**

```
python3 app.py
```

Then try:

- `What is the Calvin Cycle?`
- `quiz photosynthesis`
- `exit`

**Option B: Web chat interface (recommended)**

```
python3 server.py
```

Then open your browser to **http://127.0.0.1:5000**

You'll get a chat window where you can:

- Type questions and get answers from your notes
- Enter a topic in the quiz box and click "Generate Quiz"
- Click "Reset chat" to clear memory and start fresh

The first load takes a little longer since it builds the search index and loads the model.

## How it works

- **RAG**: Your notes are split into chunks and stored in a local Chroma
  vector database. When you ask a question, the most relevant chunks are
  found and given to the AI as context.
- **Memory**: The chat history is kept in memory so follow-up questions
  like "explain that simpler" work naturally.
- **Quiz generation**: Typing `quiz <topic>` pulls relevant notes and asks
  the AI to write multiple-choice questions from them.

## Project structure

```
study_assistant/
├── app.py                    # CLI version
├── server.py                 # Flask web server
├── rag_engine.py             # Shared RAG + memory + quiz logic
├── templates/
│   └── index.html            # Web chat frontend
├── data/
│   ├── academic_rules.txt
│   ├── admissions_eligibility.txt
│   ├── campus_discipline.txt
│   ├── college_profile.txt
│   ├── courses_seats.txt
│   ├── hostel_policies.txt
│   ├── library_rules.txt
│   ├── pasted.txt
│   ├── placement_details.txt
│   ├── sample_notes.txt
│   └── scholarships.txt
└── requirements.txt
```

## Next steps (Day 2-5 ideas)

- Add PDF support (`PyPDFLoader` instead of `TextLoader`)
- Add a "generate study plan" feature
- Support multiple documents/subjects (subject picker in the UI)
- Add persistent chat history across sessions (save to a file/DB)
- Add file upload directly from the web UI
