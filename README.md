💬 Loan Approval Chatbot

This is an interactive chatbot that answers questions about real-world loan application data. It's designed to help users understand what kind of applicants get loans approved — using a combination of retrieval-augmented generation (RAG), FAISS, and Gemini AI.

🚀 Features
Ask questions like “Do self-employed applicants get approved?” or “What is the average income of approved applicants?”

Uses real loan data to provide context-aware responses

Powered by:

FAISS for semantic search

Sentence-Transformers for embeddings

Gemini 1.5 Flash for answering

Streamlit frontend for an easy and interactive UI

🧠 How it Works
1. The app loads and processes a dataset of loan applicants.
2. Converts each row into a natural language sentence.
3. Uses vector search (FAISS) to find the most relevant data for a user’s question.
4. Sends the relevant info to Gemini AI to generate a smart, human-like answer.

<img width="1879" height="840" alt="image" src="https://github.com/user-attachments/assets/06d55535-13ff-4b93-8939-a3d447920a54" />

✅ Live App: Try the chatbot here → https://loanragchatbot-assignment08.streamlit.app/

📦 loan--RAG-chatbot/
├── app.py              # Streamlit UI
├── vectorstore.py      # FAISS index & sentence embeddings
├── chatbot.py          # Gemini querying logic
├── data/
│   └── Training_Dataset.csv
├── .env                # Your local API key (DO NOT COMMIT)
├── .gitignore
├── requirements.txt
└── README.md
