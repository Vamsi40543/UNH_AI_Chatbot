"""
File: chatbot.py
Authors: Jacob
Contributors: Vamsi Sai Krishna Valluru
Date: 02-20-25
"""
import os
import time
from flask import Flask, request, jsonify, render_template
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.memory import ConversationBufferMemory
import re
import json
from datetime import datetime
import hashlib
from langchain.chat_models import ChatOpenAI
# Flask app initialization
app = Flask(__name__)
# Route for rendering the homepage
@app.route('/')
def home():
    return render_template('index.html')
# Path to the JSON file for storing chat history and conversation memory
chat_history_file = 'chat_history.json'
conversation_memory_file = 'conversation_memory.json'
# OpenAI API key for accessing embeddings and language models
apikey = os.getenv("OPENAI_API_KEY")
# Initialize the chat history and conversation memory files if they don't exist
def initialize_chat_history_file():
    if not os.path.exists(chat_history_file):
        with open(chat_history_file, 'w') as file:
            json.dump([], file)
    if not os.path.exists(conversation_memory_file):
        with open(conversation_memory_file, 'w') as file:
            json.dump([], file)
# Load general chat history from a JSON file
def load_chat_history():
    if os.path.exists(chat_history_file):
        try:
            with open(chat_history_file, 'r') as file:
                data = json.load(file)
                return data if data else []
        except json.JSONDecodeError:
            return []
    return []
# Save general chat history to a JSON file
def save_chat_history(chat_history):
    with open(chat_history_file, 'w') as file:
        json.dump(chat_history, file, indent=4)
# Load conversation memory from a JSON file for persistent memory across sessions
def load_conversation_memory():
    if os.path.exists(conversation_memory_file):
        try:
            with open(conversation_memory_file, 'r') as file:
                data = json.load(file)
                return data if data else []
        except json.JSONDecodeError:
            return []
    return []
# Save conversation memory to a JSON file after each interaction
def save_conversation_memory(memory):
    with open(conversation_memory_file, 'w') as file:
        json.dump(memory, file, indent=4)
# Generate a consistent hash for a question
def get_question_hash(question):
    normalized_question = " ".join(question.lower().split())
    return hashlib.md5(normalized_question.encode()).hexdigest()
# Retrieve a cached response based on question hash
def get_cached_response(question_hash):
    chat_history = load_chat_history()
    for entry in chat_history:
        if entry.get("question_hash") == question_hash:
            return entry.get("bot_response")
    return None
# Save a new interaction to the chat history with question hash
def save_interaction_to_json(user_question, bot_response):
    chat_history = load_chat_history()
    question_hash = get_question_hash(user_question)
    new_chat = {
        "user_question": user_question,
        "bot_response": bot_response,
        "timestamp": datetime.now().isoformat(),
        "question_hash": question_hash
    }
    # Check if question exists and update if necessary
    for entry in chat_history:
        if entry.get("question_hash") == question_hash:
            entry.update(new_chat)
            break
    else:
        chat_history.append(new_chat)
    save_chat_history(chat_history)
# Extract text and table data from PDF using pdfplumber
def extract_text_from_pdf(pdf_path):
    text = ''
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ''  # Extract text from page
            text += page_text + "\n"  # Add newline for separation
            tables = page.extract_table()
            if tables:
                text += "\n\n" + "\n".join(
                    ["\t".join([str(cell) if cell is not None else '' for cell in row]) for row in tables if row]
                ) + "\n"
    return text
# Handle multiple PDFs in a directory
def extract_texts_from_multiple_pdfs(pdf_directory):
    documents = []
    for pdf_file in os.listdir(pdf_directory):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(pdf_directory, pdf_file)
            pdf_text = extract_text_from_pdf(pdf_path)
            documents.append(Document(page_content=pdf_text, metadata={"source": pdf_file}))
    return documents
# Directory containing PDFs to be processed

pdf_directory = ''# place your path

documents = extract_texts_from_multiple_pdfs(pdf_directory)
# Split documents into chunks for better context management
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200,
    separators=["\n\n", "\n", " "]
)

def calculate_similarity(vec1, vec2, strict_threshold=0.75, min_magnitude=0.1):
    """
    Calculate cosine similarity between two vectors with strict thresholds.
    
    Args:
        vec1 (list): First vector
        vec2 (list): Second vector
        strict_threshold (float): Minimum similarity required to return a non-zero value
        min_magnitude (float): Minimum magnitude threshold to consider relevant
        
    Returns:
        float: Cosine similarity if above threshold and magnitudes are significant, 0 otherwise
    """
    # Calculate dot product
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    
    # Calculate magnitudes
    magnitude1 = sum(a ** 2 for a in vec1) ** 0.5
    magnitude2 = sum(b ** 2 for b in vec2) ** 0.5
    
    # Return 0 if either magnitude is too small or zero
    if magnitude1 < min_magnitude or magnitude2 < min_magnitude or magnitude1 * magnitude2 == 0:
        return 0
    
    # Calculate cosine similarity
    similarity = dot_product / (magnitude1 * magnitude2)
    
    # Apply strict threshold - only return similarity if it's high enough
    if similarity < strict_threshold:
        return 0
    
    return similarity

def create_contact_knowledge_base():
    # Define contacts with descriptions of their areas of responsibility
    contacts = [
        {
            "name": "Professor Timothy Finan",
            "email": "timothy.finan@unh.edu",
            "description": "Program details, course information, curriculum questions, degree requirements, program coordination for Cybersecurity Engineering and IT programs"
        },
        {
            "name": "Professor Karen Jin",
            "email": "karen.jin@unh.edu",
            "description": "Internship opportunities, COMP 891, COMP 893, COMP 898, external internships, team project internships, master's project, internship coordination"
        },
        {
            "name": "Christine Rousseau",
            "email": "christine.rousseau@unh.edu",
            "description": "Orientation details, onboarding, welcome activities, new student information sessions"
        },
        {
            "name": "Health & Wellness",
            "email": "health@unh.edu",
            "description": "Immunization requirements, vaccines, health insurance, SHBP, tuberculosis testing, health records, medical requirements"
        },
        {
            "name": "Student Accounts",
            "email": "Student.Accounts@unh.edu",
            "description": "Tuition payments, billing, fees, financial questions, payment plans, scholarships, refunds"
        },
        {
            "name": "OISS",
            "email": "oiss@unh.edu",
            "description": "Visa information, I-20, international student requirements, CPT, OPT, SEVIS, immigration status"
        }
    ]
    
    # Generate embeddings for each contact description
    contact_embeddings = []
    for contact in contacts:
        embedding = embeddings.embed_query(contact["description"])
        contact_embeddings.append({
            "contact": contact,
            "embedding": embedding
        })
    
    return contact_embeddings

def find_relevant_contact(query, contact_embeddings):
    # Embed the user query
    query_embedding = embeddings.embed_query(query)
    
    # Find the closest match
    max_similarity = -1
    most_relevant_contact = None
    
    for item in contact_embeddings:
        similarity = calculate_similarity(query_embedding, item["embedding"])
        
        if similarity > max_similarity:
            max_similarity = similarity
            most_relevant_contact = item["contact"]
    
    # Only return a contact if the similarity is above a threshold (increased from 0.5)
    if max_similarity > 0.75:  # Higher threshold for stricter matching
        return most_relevant_contact
    return None

def create_links_knowledge_base():
    """
    Create a knowledge base of standardized links with descriptions of their content.
    Returns a list of links with their embeddings for semantic search.
    """
    # Define links with descriptions of their content
    links = [
        {
            "title": "MS in Cybersecurity Engineering Program",
            "url": "https://manchester.unh.edu/program/ms/cybersecurity-engineering",
            "description": "Cybersecurity engineering program, security courses, cyber program details, MS cybersecurity curriculum, cybersecurity degree requirements"
        },
        {
            "title": "MS in Information Technology Program",
            "url": "https://manchester.unh.edu/program/ms/information-technology",
            "description": "Information Technology program, IT courses, MS IT curriculum, information technology degree, computer science, software development"
        },
        {
            "title": "Faculty Directory",
            "url": "https://mobile.unh.edu/UNHMobile/directory/facultystaff.jsp",
            "description": "Faculty directory, professor contact information, staff directory, instructor information, faculty profiles, teacher contact"
        },
        {
            "title": "Health & Wellness",
            "url": "https://www.unh.edu/health",
            "description": "Health services, wellness resources, medical assistance, health center, counseling, mental health, physical health"
        },
        {
            "title": "Student Health Insurance",
            "url": "https://www.unh.edu/health/student-health-insurance",
            "description": "Health insurance, SHBP, student health benefits plan, medical coverage, health plan, insurance requirements"
        },
        {
            "title": "Immunization Form",
            "url": "https://www.unh.edu/health/sites/default/files/media/2022-08/unh-health-wellness-immunization-form-2022.pdf",
            "description": "Immunization requirements, vaccination records, required vaccines, MMR, meningococcal, tdap, varicella, TB testing"
        },
        {
            "title": "Health Benefits Plan Brochure",
            "url": "https://www.unh.edu/health/sites/default/files/media/2024-06/unh-shbp-brochure-2024-2025_final.pdf",
            "description": "Health insurance details, coverage information, insurance benefits, health plan costs, medical coverage details"
        },
        {
            "title": "Office of International Students and Scholars (OISS)",
            "url": "https://www.unh.edu/global/international-students",
            "description": "OISS, international students, visa information, I-20, CPT, OPT, SEVIS, immigration status, international student support"
        },
        {
            "title": "Tuition and Fees",
            "url": "https://www.unh.edu/business-services/tuition-fees/unh-manchester-graduate-school",
            "description": "Tuition costs, fees, payment information, graduate tuition rates, credit hour costs, billing information"
        },
        {
            "title": "Student Accounts",
            "url": "https://www.unh.edu/business-services/student-accounts",
            "description": "Student billing, payment methods, account information, billing statements, payment deadlines, financial holds"
        },
        {
            "title": "Housing",
            "url": "https://www.apartments.com/manchester-nh",
            "description": "Apartments near campus, rental listings, housing search, rental properties, places for rent in Manchester"
        },
        {
            "title": "Transportation - Boston Express Bus",
            "url": "https://www.bostonexpressbus.com",
            "description": "Bus service, transportation to Boston, Logan airport transportation, commuting options, bus schedules"
        },
        {
            "title": "Transportation - Concord Coach",
            "url": "https://concordcoachlines.com",
            "description": "Bus service, Transportation from boston Airport, travel to New Hampshire, Logan airport to Manchester, bus routes"
        },
        {
            "title": "Manchester Transit Authority",
            "url": "https://mtabus.org",
            "description": "Local bus service, public transportation in Manchester, city bus routes, local transit options"
        },
        {
            "title": "WebCat Student Portal",
            "url": "https://my.unh.edu",
            "description": "Student portal, course registration, academic records, class schedule, grade viewing, account management"
        },
        {
            "title": "Campus Safety",
            "url": "https://manchester.unh.edu/student-experience/campus-safety",
            "description": "Security information, emergency services, campus police, security resources, safety protocols, emergency contacts"
        },
        {
            "title": "Student Wellness",
            "url": "https://manchester.unh.edu/academics/academic-services/student-wellness",
            "description": "Wellness resources, mental health support, counseling services, health promotion, stress management"
        },
        {
            "title": "UNH Directory",
            "url": "https://mobile.unh.edu/UNHMobile/directory/facultystaff.jsp",
            "description": "University directory, contact search, find faculty, staff lookup, employee contact information"
        }
    ]
    
    # Generate embeddings for each link description
    link_embeddings = []
    for link in links:
        embedding = embeddings.embed_query(link["description"])
        link_embeddings.append({
            "link": link,
            "embedding": embedding
        })
    
    return link_embeddings

def find_relevant_links(query, link_embeddings, max_links=3, similarity_threshold=0.8):
    """
    Find the most relevant links for a given query using semantic similarity.
    Returns up to max_links number of links that exceed the similarity threshold.
    Increased default threshold from 0.6 to 0.8 for stricter matching.
    """
    # Embed the user query
    query_embedding = embeddings.embed_query(query)
    
    # Calculate similarity for all links
    links_with_scores = []
    
    for item in link_embeddings:
        similarity = calculate_similarity(query_embedding, item["embedding"], strict_threshold=0.5)
        links_with_scores.append({
            "link": item["link"],
            "similarity": similarity
        })
    
    # Sort by similarity (highest first)
    links_with_scores.sort(key=lambda x: x["similarity"], reverse=True)
    
    # Filter by threshold and limit to max_links
    relevant_links = [
        item["link"] for item in links_with_scores 
        if item["similarity"] > similarity_threshold
    ][:max_links]
    
    return relevant_links

def add_relevant_links_to_response(user_question, bot_response):
    """
    Enhance the bot response with relevant links based on the user's question.
    Only adds links that are semantically relevant to the question.
    """
    # Skip if the response already contains links
    if "<a href=" in bot_response:
        return bot_response
    
    # Find relevant links
    relevant_links = find_relevant_links(user_question, link_embeddings)
    
    # If no relevant links found, return the original response
    if not relevant_links:
        return bot_response
    
    # Add relevant links section to the response
    links_section = "\n\nRelevant resources:"
    for link in relevant_links:
        links_section += f"\n• {link['title']}: <a href=\"{link['url']}\" target=\"_blank\">{link['url']}</a>"
        break
    
    return bot_response + links_section


# Custom function to split documents based on weeks
def custom_split_documents_by_weeks(documents):
    chunks = []
    for doc in documents:
        if "<TABLE_START>" in doc.page_content:
            week_sections = re.split(r"(Week \d+)", doc.page_content)
            current_week = None
            for part in week_sections:
                week_match = re.match(r"Week \d+", part)
                if week_match:
                    current_week = part.strip()
                elif current_week:
                    chunks.append(Document(page_content=f"{current_week}\n{part.strip()}", metadata=doc.metadata))
        else:
            chunked_texts = text_splitter.split_text(doc.page_content)
            for chunk in chunked_texts:
                chunks.append(Document(page_content=chunk, metadata=doc.metadata))
    return chunks

# Split documents into chunks with table handling by weeks
texts = custom_split_documents_by_weeks(documents)
# Load OpenAI Embeddings for Semantic Search
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=apikey)
# Create Chroma Index for Vector Store using OpenAI Embeddings
persist_directory = 'db'
if os.path.exists(persist_directory):
    os.system(f"rm -rf {persist_directory}")
db = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory=persist_directory)
# Create a retriever with OpenAI embeddings
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":5})
#  Set up conversational memory and load past memory
previous_memory = load_conversation_memory()
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)
for item in previous_memory:
    # Load past messages into memory
    memory.chat_memory.add_user_message(item["user"])
    memory.chat_memory.add_ai_message(item["assistant"])
# Define the custom prompt template for friendly, conversational tone

PROMPT_TEMPLATE = """
You are a friendly and knowledgeable chatbot designed to assist users with questions related to the UNH International Orientation and the UNH Computing degrees - M.S Information Technology or M.S. Cybersecurity Engineering. Your goal is to provide accurate and helpful information in a conversational manner.
For greeting:
1. Only respond with a greeting if the user's message is a greeting or they're starting a new conversation. For example:
   "Hello! I am the ByteCat UNHM Advising Chatbot. How may I assist you today?"
   Otherwise, respond directly to their question.
2. For subsequent messages in the same conversation, focus on answering the specific question without repeating greetings.
3. For out-of-context questions:
   Respond with: "Sorry, I don't have information about that question. Feel free to ask me anything related to the UNH Computing degrees -  M.S Information Technology or M.S. Cybersecurity Engineering."
4. Answer Format:
  ● Keep responses brief and factual.
  ● Avoid using bold asterisks (`*`) or styled formatting in your answers. Present all information in plain text.
  ● Maintain a professional and informative tone.
  ● Ensure proper spacing between bullet points by using line breaks (`\n\n`) between each item.
  ● Use bullet points properly on separate lines, not in a paragraph format.
5. Response Guidelines:
  ● Keep responses brief and factual.
  ● Format answers with clear, well-spaced bullet points.
Keep responses very brief with 3-5 bullet points maximum. Only include contact information if directly relevant to the question topic.
Previous interactions with the user:
{chat_history}
Context related to the question:
{context}
Current Question:
{question}
Answer:
provide the most relevant information first, and offer to provide further details if needed.
"""

# Initialize the open AI LLM Model
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    openai_api_key=apikey,
    temperature=0.5,
    top_p=1,
    max_tokens=250
)
from langchain.schema import HumanMessage
# API route to handle user questions
import re
# Initialize contact embeddings when the app starts
contact_embeddings = create_contact_knowledge_base()
link_embeddings = create_links_knowledge_base()
@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    user_question = data.get('message', '').strip()  # Ensure a default empty string
    
    # Ensure user_question is a string
    if not isinstance(user_question, str) or not user_question:
        return jsonify({"error": "Invalid question format."}), 400

    # Handle greetings explicitly
    if user_question.lower() in ['hi', 'hello']:
        return jsonify({
            "response": "Hello! I am the ByteCat UNHM Advising Chatbot. How may I assist you today?",
            "retrieval_context": [],
            "sources_used": []
        })
    
    # Handle specific queries about free food
    if user_question.lower() in ['free food', 'where do i get free food', 'where can i get free food']:
        return jsonify({
            "response": "UNH Manchester also has a pantry available for students in need.\n☞ Check with the Student Involvement Office or campus event calendars for upcoming events offering free food.\n☞ Free food may be available at certain campus events, orientation, or student organization meetings.\n☞ For more specific information or assistance, feel free to ask!",
            "retrieval_context": [],
            "sources_used": []
        })

    #Handle specific queries about course schedule
    if user_question.lower() in ['course schedule', 'can you tell me my schedule', 'what is my schedule', 'tell me my schedule']:
        return jsonify({
            "response":"Once registered to your desired course, you can get the course schedule on WebCat. \n☞ Log in with your UNH credentials.\n☞ Click Registration -Registration Information.\n☞ Click View Registration Information.\n☞ Select the applicable term.\n☞ This will show you your schedule for the selected term.\n☞ If you have any questions about your schedule, please contact the Student Accounts office.",
            "retrieval_context": [],
            "sources_used": []
        })
    
    # Generate question hash for caching
    question_hash = get_question_hash(user_question)

    # Check for a cached response for identical questions
    cached_response = get_cached_response(question_hash)
    if cached_response:
        return jsonify({
            "response": cached_response, 
            "retrieval_context": [],
            "sources_used": []
        })
    
    # Retrieve relevant context directly using similarity_search
    relevant_docs = db.similarity_search(user_question, **retriever.search_kwargs)[:2]
    retrieval_context = [doc.page_content for doc in relevant_docs]
    context = "\n".join(retrieval_context[:4])
    
    # Extract source information for debugging
    sources_used = []
    sources_info = []
    for i, doc in enumerate(relevant_docs):
        source = doc.metadata.get('source', 'unknown')
        content_preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
        
        # Add to sources list if not already included
        if source not in sources_used:
            sources_used.append(source)
        
        # Create detailed source info for logging
        sources_info.append({
            "source": source,
            "content_preview": content_preview
        })
        
        # Log source information
        print(f"Source {i}: {source}")
        print(f"Content excerpt: {content_preview}")
    
    # Load conversation memory to populate chat history
    conversation_memory_data = load_conversation_memory()
    
    # Dynamically populate chat history from conversation memory
    chat_history = "\n".join([
        f"User: {item['user']}\nAssistant: {item['assistant']}"
        for item in conversation_memory_data
    ])
    
    # Use the custom prompt template
    prompt_text = PROMPT_TEMPLATE.format(
        context=context,
        chat_history=chat_history,
        question=user_question
    )
    
    start_time = time.time()
    
    # Generate a new response using the formatted prompt with HumanMessage and invoke
    response = llm.invoke([HumanMessage(content=prompt_text)], max_tokens=350, temperature=0)
    print(f"API Response Time: {time.time() - start_time} seconds")
    
    def make_links_clickable(text):
        url_pattern = r'(https?://[^\s]+?)([.,;:!?)\]\'"]*)(?=\s|$)'
    
        # Replace URLs with HTML anchor tags, preserving any trailing punctuation outside the link
        return re.sub(url_pattern, r'<a href="\1" target="_blank">\1</a>\2', text)
    
    # Format the response content
    answer = response.content.replace("\n- ", "\n☞ ") if response else "No answer found"
    answer = make_links_clickable(answer)

    # Add source information footer to the response
    #source_footer = "\n\n<em>Information sourced from: " + ", ".join(sources_used) + "</em>"
    #answer += source_footer

    # Then add relevant links semantically matched to the query (if none are present)
    answer = add_relevant_links_to_response(user_question, answer)

    # Find and add relevant contact information
    relevant_contact = find_relevant_contact(user_question, contact_embeddings)

    # Add contact information if relevant and not already included
    if relevant_contact:
        # Check if this contact's email is already mentioned in the response
        if relevant_contact["email"] and relevant_contact["email"] not in answer:
            contact_info = f"\n\nFor more information about this topic, please contact {relevant_contact['name']} at <strong>{relevant_contact['email']}</strong>."
            answer += contact_info

    # Save the interaction to conversation memory (limit to last 5 interactions)
    conversation_memory_data.append({"user": user_question, "assistant": answer})
    conversation_memory_data = conversation_memory_data[-5:]  # Keep only the last 5 interactions
    save_conversation_memory(conversation_memory_data)
    
    # Save the interaction to chat history
    save_interaction_to_json(user_question, answer)
    
    # Return the response with retrieval context and source information
    return jsonify({
        "response": answer, 
        "retrieval_context": retrieval_context,
        "sources_used": sources_used
    })

if __name__ == '__main__':
    initialize_chat_history_file()
    app.run(debug=True, host='0.0.0.0', port=80)
