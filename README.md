#  Bytecat - UNHM Advising Chatbot

## What is This Project?
The **Bytecat - UNHM Advising Chatbot** is a helpful tool for graduate students at the University of New Hampshire (UNH).  
It can answer questions about:
- **Courses** – like descriptions, prerequisites, and credits.
- **Orientation** – such as deadlines, policies, contacts, and where to find resources.

It uses powerful AI (OpenAI GPT-3.5 Turbo) to give smart and helpful answers in a conversation style.

---

## System Architecture (Step-by-Step Flow)

Here’s how the Bytecat - UNHM Advising chatbot works behind the scenes:

### 1. Data Ingestion
- **PDF Documents** (like course catalogs and orientation guides) are added to the system.
- The PDFs go through **PDF Processing** to extract **text and tables**.

### 2. Preprocessing
- The extracted text is **chunked** into smaller, manageable pieces.
- These chunks are converted into **embeddings** using `text-embedding-ada-002` from OpenAI.
- The embeddings are stored in a **Chroma Vector Store** for fast retrieval.

### 3. When a User Asks a Question
- The chatbot first **checks if the same question was asked before** using the **chat history or cache**.

  - **If found in cache:** It returns the **cached response**.
  - **If not found in cache:** It does a **semantic search** in the Chroma database to find relevant chunks.

### 4. Answer Generation
- It uses the matching chunks to create a **contextual prompt**.
- That prompt is sent to **OpenAI GPT-4 Turbo**, which generates a **smart and conversational reply**.

### 5. Memory & Response
- The response is **saved in the chat history** for future use (context + caching).
- Finally, the **answer is returned to the user**.

---

## 🗺️ Flow diagram

![alt text](https://github.com/UNHM-TEAM-PROJECT/Spring2025-Team-Rivals/blob/main/Rivals_Chatbot/static/image.png)


---

## Tools and Technologies

- **Flask** – runs the web app  
- **LangChain** – connects everything together  
- **Chroma** – stores the processed data  
- **OpenAI GPT-4 Turbo** – the AI that gives smart replies  
- **pdfplumber** – extracts text from PDFs  
- **Python 3.8+** – programming language used

---

## Folder and File Structure

Here’s what’s inside the project folder:

| File/Folder                 | What It’s For                                     |
|----------------------------|---------------------------------------------------|
| `automated_testing/`       | Scripts to test how well the chatbot performs     |
| `data/`                    | Where you put the course and orientation PDFs     |
| `static/`                  | CSS or JS files for the website interface         |
| `templates/`               | HTML files used by Flask                          |
| `chatbot.py`               | The main chatbot code                             |
| `requirements.txt`         | List of required Python libraries                 |
| `README.md`                | This overview file                                |
| `INSTALLATION_GUIDE.md`    | Step-by-step setup instructions                   |

- Put all your PDFs in the `data/` folder  
- Update your API key and file paths in `chatbot.py` before running the app

---

## How to Set It Up

To install and run everything, follow the steps in [INSTALLATION_GUIDE.md](https://github.com/UNHM-TEAM-PROJECT/Spring2025-Team-Rivals/blob/main/Installation_Guide.md).
  
It covers:
- What you need before starting
- How to run it on your computer
- How to test it
- How to put it on AWS (Amazon cloud)

---
