import streamlit as st
import os
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"
from transformers import AutoTokenizer, AutoModelForCausalLM
from pymongo import MongoClient
from datetime import datetime
import torch
import bcrypt
import uuid
import time
import requests
import base64
import numpy as np
import re
import json
from typing import List, Dict, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------
# Configuration
st.set_page_config(
    page_title="DexiBot - AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# MongoDB Configuration
MONGODB_URI = "mongodb+srv://sayanmaity8001:sayan@cluster0.qwj6tix.mongodb.net/digibot"
DB_NAME = "personalized_chat_asst"
USERS_COLLECTION = "users"
EMPLOYEE_CHATS_COLLECTION = "employee_chats"

# Model paths for Tiny Llama
MODEL_PATH = "./saved_tinyllama_model"

# Hugging Face Configuration
HF_TOKEN = "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"
HF_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

# ----------------------------
# Session state initialization
def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        'authenticated': False,
        'username': None,
        'user_id': None,
        'model': None,
        'tokenizer': None,
        'chat_history': [],
        'show_signup': False,
        'model_loaded': False
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# ----------------------------
# Database Connection
@st.cache_resource
def init_mongodb():
    """Initialize MongoDB connection with error handling"""
    try:
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        db = client[DB_NAME]
        # Test connection
        client.admin.command('ping')
        logger.info("MongoDB connection successful")
        return db
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        st.error(f"Database connection failed: {e}")
        return None

# ----------------------------
# Model Loading Functions
def check_model_files(dir_path: str) -> bool:
    """Check if directory contains required model files"""
    if not os.path.isdir(dir_path):
        return False
    
    required_files = ["config.json"]
    files = set(os.listdir(dir_path))
    
    # Check for config.json
    if "config.json" not in files:
        return False
    
    # Check for model weights
    weight_files = [
        "pytorch_model.bin",
        "model.safetensors"
    ]
    
    if any(f in files for f in weight_files):
        return True
    
    # Check for sharded model files
    sharded_files = [f for f in files if f.startswith("pytorch_model-") and f.endswith(".bin")]
    return len(sharded_files) > 0

@st.cache_resource
def load_digibot_model():
    """Load Hugging Face model and tokenizer from local path"""
    if not check_model_files(MODEL_PATH):
        error_msg = f"Model files not found in '{MODEL_PATH}'. Please ensure the model is properly saved."
        logger.error(error_msg)
        st.error(error_msg)
        return None, None
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH, 
            local_files_only=True,
            trust_remote_code=False
        )
        
        # Set pad token if not exists
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        logger.info("Tokenizer loaded successfully")
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            device_map="cpu",
            local_files_only=True,
            trust_remote_code=False
        )
        
        model.eval()
        logger.info("Model loaded successfully")
        
        return model, tokenizer
        
    except Exception as e:
        error_msg = f"Failed to load model: {e}"
        logger.error(error_msg)
        st.error(error_msg)
        return None, None

# ----------------------------
# Authentication Functions
def hash_password(password: str) -> bytes:
    """Hash password using bcrypt"""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

def verify_password(password: str, hashed: bytes) -> bool:
    """Verify password against hash"""
    return bcrypt.checkpw(password.encode('utf-8'), hashed)

def create_user(db, username: str, email: str, password: str, full_name: str) -> Tuple[bool, Optional[str], str]:
    """Create new user account"""
    try:
        users_collection = db[USERS_COLLECTION]
        
        # Check if user already exists
        existing_user = users_collection.find_one({
            "$or": [{"username": username}, {"email": email}]
        })
        
        if existing_user:
            return False, None, "Username or email already exists"
        
        # Create new user
        user_id = str(uuid.uuid4())
        user_doc = {
            "user_id": user_id,
            "username": username,
            "email": email,
            "full_name": full_name,
            "password_hash": hash_password(password),
            "created_at": datetime.now(),
            "last_login": None,
            "is_active": True
        }
        
        users_collection.insert_one(user_doc)
        logger.info(f"User created successfully: {username}")
        return True, user_id, "User created successfully"
        
    except Exception as e:
        logger.error(f"Error creating user: {e}")
        return False, None, f"Error creating user: {e}"

def authenticate_user(db, username: str, password: str) -> Tuple[bool, Optional[Dict]]:
    """Authenticate user credentials"""
    try:
        users_collection = db[USERS_COLLECTION]
        user = users_collection.find_one({
            "username": username, 
            "is_active": True
        })
        
        if user and verify_password(password, user['password_hash']):
            # Update last login
            users_collection.update_one(
                {"_id": user['_id']},
                {"$set": {"last_login": datetime.now()}}
            )
            logger.info(f"User authenticated successfully: {username}")
            return True, user
        else:
            logger.warning(f"Authentication failed for user: {username}")
            return False, None
            
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        return False, None

# ----------------------------
# Embedding Functions
def get_hf_embedding(text: str, max_retries: int = 3) -> Optional[np.ndarray]:
    """Get embedding from Hugging Face Inference API (router endpoint) with retry logic."""
    if not text.strip():
        return None
    API_URL = "https://router.huggingface.co/hf-inference/models/BAAI/llm-embedder/pipeline/feature-extraction"
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
    }
    for attempt in range(max_retries):
        try:
            payload = {"inputs": text}
            response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
            if response.status_code == 200:
                data = response.json()
                # Convert to numpy array and get mean pooling
                embedding = np.array(data).mean(axis=0)
                return embedding.astype(np.float32)
            elif response.status_code == 503:
                # Model loading, wait and retry
                wait_time = min(2 ** attempt, 10)
                time.sleep(wait_time)
                continue
            else:
                logger.error(f"HF API error {response.status_code}: {response.text}")
                return None
        except Exception as e:
            logger.error(f"Error calling HF API (attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                return None
            time.sleep(1)
    return None

# ----------------------------
# Text Processing Functions
def chunk_text(text: str, max_chars: int = 1000, overlap: int = 100) -> List[str]:
    """Chunk text with overlap for better context preservation"""
    if len(text) <= max_chars:
        return [text]
    
    # Split by sentences first
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # If adding this sentence would exceed max_chars
        if len(current_chunk) + len(sentence) > max_chars:
            if current_chunk:
                chunks.append(current_chunk.strip())
                # Start new chunk with overlap
                words = current_chunk.split()
                overlap_words = words[-overlap//10:] if len(words) > overlap//10 else []
                current_chunk = " ".join(overlap_words) + " " + sentence
            else:
                # Single sentence is too long, force split
                chunks.append(sentence[:max_chars])
                current_chunk = sentence[max_chars:]
        else:
            current_chunk += " " + sentence if current_chunk else sentence
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return [chunk for chunk in chunks if chunk.strip()]

# ----------------------------
# OCR Functions
def ocr_extract_text(image_path: str, api_key: str = "22308f269288957") -> str:
    """Extract text from image using OCR.space API"""
    try:
        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode()
        
        url = 'https://api.ocr.space/parse/image'
        payload = {
            'apikey': api_key,
            'base64Image': f'data:image/jpeg;base64,{image_data}',
            'language': 'eng',
            'detectOrientation': 'true',
            'scale': 'true',
            'OCREngine': '2',
            'isTable': 'false'
        }
        
        response = requests.post(url, data=payload, timeout=60)
        result = response.json()
        
        if result.get('IsErroredOnProcessing', True):
            error_msg = result.get('ErrorMessage', 'Unknown OCR error')
            logger.error(f"OCR Error: {error_msg}")
            return f"OCR Error: {error_msg}"
        
        if result.get('ParsedResults') and len(result['ParsedResults']) > 0:
            extracted_text = result['ParsedResults'][0].get('ParsedText', "")
            return extracted_text.strip()
        else:
            return "No text found in image"
            
    except Exception as e:
        logger.error(f"OCR processing failed: {e}")
        return f"OCR processing failed: {str(e)}"

# ----------------------------
# Document Management Functions
def save_user_document(db, user_id: str, title: str, content: str, description: str = "") -> Optional[str]:
    """Save document and create embeddings for chunks"""
    if not content.strip():
        st.error("Cannot save empty document")
        return None
    
    try:
        # Save document metadata
        docs_collection = db[f"user_docs_{user_id}"]
        doc_id = str(uuid.uuid4())
        
        doc = {
            "doc_id": doc_id,
            "user_id": user_id,
            "title": title,
            "content": content,
            "description": description,
            "created_at": datetime.now()
        }
        
        docs_collection.insert_one(doc)
        logger.info(f"Document saved: {title}")
        
        # Create chunks and embeddings
        chunks = chunk_text(content, max_chars=1000)
        if not chunks:
            logger.warning("No chunks created from document")
            return doc_id
        
        chunks_collection = db[f"user_chunks_{user_id}"]
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        successful_chunks = 0
        total_chunks = len(chunks)
        
        for idx, chunk in enumerate(chunks):
            progress = (idx + 1) / total_chunks
            progress_bar.progress(progress)
            status_text.text(f"Processing chunk {idx + 1}/{total_chunks}...")
            
            # Get embedding
            embedding = get_hf_embedding(chunk)
            if embedding is None:
                logger.warning(f"Failed to embed chunk {idx + 1}")
                continue
            
            # Save chunk with embedding
            chunk_doc = {
                "doc_id": doc_id,
                "chunk_index": idx,
                "text": chunk,
                "embedding": embedding.tolist(),
                "created_at": datetime.now()
            }
            
            try:
                chunks_collection.insert_one(chunk_doc)
                successful_chunks += 1
            except Exception as e:
                logger.error(f"Failed to save chunk {idx + 1}: {e}")
        
        progress_bar.empty()
        status_text.empty()

        if successful_chunks > 0:
            st.success(f"Document saved with {successful_chunks}/{total_chunks} chunks embedded successfully!")
        else:
            st.warning("Document saved but no chunks were embedded successfully. Please check your embedding API or try again later.")

        return doc_id
        
    except Exception as e:
        logger.error(f"Error saving document: {e}")
        st.error(f"Error saving document: {e}")
        return None

def get_user_documents(db, user_id: str) -> List[Dict]:
    """Retrieve all documents for a user"""
    try:
        docs_collection = db[f"user_docs_{user_id}"]
        docs = list(docs_collection.find({"user_id": user_id}))
        # Sort by creation date (newest first)
        docs.sort(key=lambda x: x.get("created_at", datetime.min), reverse=True)
        return docs
    except Exception as e:
        logger.error(f"Error loading user documents: {e}")
        return []

def delete_user_document(db, user_id: str, doc_id: str) -> bool:
    """Delete document and all its chunks"""
    try:
        docs_collection = db[f"user_docs_{user_id}"]
        chunks_collection = db[f"user_chunks_{user_id}"]
        
        # Delete document
        docs_result = docs_collection.delete_many({"doc_id": doc_id})
        chunks_result = chunks_collection.delete_many({"doc_id": doc_id})
        
        logger.info(f"Deleted document {doc_id}: {docs_result.deleted_count} docs, {chunks_result.deleted_count} chunks")
        return docs_result.deleted_count > 0
        
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        return False

def retrieve_similar_chunks(db, user_id: str, query: str, top_k: int = 3) -> List[Dict]:
    """Retrieve most similar document chunks for a query"""
    # Get query embedding
    query_embedding = get_hf_embedding(query)
    if query_embedding is None:
        logger.warning("Failed to embed query")
        return []
    
    try:
        chunks_collection = db[f"user_chunks_{user_id}"]
        chunks = list(chunks_collection.find({}, {"text": 1, "embedding": 1, "doc_id": 1}))
        
        if not chunks:
            return []
        
        # Calculate similarities
        similarities = []
        for chunk in chunks:
            if "embedding" not in chunk or not chunk["embedding"]:
                continue
            
            chunk_embedding = np.array(chunk["embedding"], dtype=np.float32)
            
            # Cosine similarity
            dot_product = np.dot(query_embedding, chunk_embedding)
            norm_query = np.linalg.norm(query_embedding)
            norm_chunk = np.linalg.norm(chunk_embedding)
            
            if norm_query > 0 and norm_chunk > 0:
                similarity = dot_product / (norm_query * norm_chunk)
                similarities.append({
                    "text": chunk["text"],
                    "similarity": float(similarity),
                    "doc_id": chunk.get("doc_id", "")
                })
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        return similarities[:top_k]
        
    except Exception as e:
        logger.error(f"Error retrieving similar chunks: {e}")
        return []

# ----------------------------
# Chat Functions
def generate_digibot_response(model, tokenizer, query: str, context: str = None, max_length: int = 200, temperature: float = 0.7) -> str:
    """Generate response using the local model with a single-turn, natural prompt. Use 'Answer:' as delimiter and return only the answer and don't create any followup questions on your own."""
    try:
        # Construct prompt
        if context:
            prompt = f"You are DexiBot, a friendly and natural-sounding AI assistant. Use the provided context to answer the user's question below. If the context doesn't contain enough information, answer naturally and honestly, and say if you don't know.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        else:
            prompt = f"You are DexiBot, a friendly and natural-sounding AI assistant. Answer the user's question below. If you don't know the answer, say so naturally.\n\nQuestion: {query}\n\nAnswer:"
        logger.info(f"Prompt sent to model:\n{prompt}")
        # Tokenize input
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
            padding=True
        )
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=inputs["input_ids"].shape[1] + max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
                top_p=0.9
            )
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Only return the part after 'Answer:'
        if 'Answer:' in response:
            answer = response.split('Answer:', 1)[-1].strip()
        else:
            answer = response.strip()
        return answer if answer else "I'm sorry, I couldn't generate a response. Please try again."
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return f"I encountered an error while generating a response: {str(e)}"

def save_chat_message(db, user_id: str, message: str, response: str) -> bool:
    """Save chat message to database"""
    try:
        chats_collection = db[EMPLOYEE_CHATS_COLLECTION]
        chat_doc = {
            "chat_id": str(uuid.uuid4()),
            "user_id": user_id,
            "message": message,
            "response": response,
            "timestamp": datetime.now()
        }
        chats_collection.insert_one(chat_doc)
        return True
    except Exception as e:
        logger.error(f"Error saving chat: {e}")
        return False

def load_user_chat_history(db, user_id: str, limit: int = 50) -> List[Dict]:
    """Load user's chat history"""
    try:
        chats_collection = db[EMPLOYEE_CHATS_COLLECTION]
        chats = chats_collection.find(
            {"user_id": user_id}
        ).sort("timestamp", -1).limit(limit)
        return list(reversed(list(chats)))
    except Exception as e:
        logger.error(f"Error loading chat history: {e}")
        return []

# ----------------------------
# UI Functions
def show_login_form(db):
    """Display login form"""
    st.title("🤖 DexiBot - Your Personal AI Assistant")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.container():
            st.markdown("### Welcome Back!")
            
            with st.form("login_form"):
                username = st.text_input("Username", placeholder="Enter your username")
                password = st.text_input("Password", type="password", placeholder="Enter your password")
                
                col_login, col_signup = st.columns(2)
                
                with col_login:
                    login_button = st.form_submit_button("🔑 Login", use_container_width=True)
                
                with col_signup:
                    signup_button = st.form_submit_button("📝 Sign Up", use_container_width=True)
                
                if login_button:
                    if username and password:
                        with st.spinner("Logging in..."):
                            success, user = authenticate_user(db, username, password)
                            if success:
                                st.session_state.authenticated = True
                                st.session_state.username = user['username']
                                st.session_state.user_id = user['user_id']
                                st.success("Login successful!")
                                st.rerun()
                            else:
                                st.error("Invalid username or password")
                    else:
                        st.error("Please enter both username and password")
                
                if signup_button:
                    st.session_state.show_signup = True
                    st.rerun()

def show_signup_form(db):
    """Display signup form"""
    st.title("🤖 Create Your DexiBot Account")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.container():
            st.markdown("### Join DexiBot Today!")
            
            with st.form("signup_form"):
                full_name = st.text_input("Full Name", placeholder="Enter your full name")
                email = st.text_input("Email", placeholder="Enter your email address")
                username = st.text_input("Username", placeholder="Choose a username")
                password = st.text_input("Password", type="password", placeholder="Create a password")
                confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm your password")
                
                col_create, col_back = st.columns(2)
                
                with col_create:
                    create_button = st.form_submit_button("✨ Create Account", use_container_width=True)
                
                with col_back:
                    back_button = st.form_submit_button("← Back to Login", use_container_width=True)
                
                if create_button:
                    if all([full_name, email, username, password, confirm_password]):
                        if password != confirm_password:
                            st.error("Passwords do not match")
                        elif len(password) < 6:
                            st.error("Password must be at least 6 characters long")
                        else:
                            with st.spinner("Creating account..."):
                                success, user_id, message = create_user(db, username, email, password, full_name)
                                if success:
                                    st.success(message)
                                    # Auto-login after successful registration
                                    st.session_state.authenticated = True
                                    st.session_state.username = username
                                    st.session_state.user_id = user_id
                                    st.session_state.show_signup = False
                                    st.rerun()
                                else:
                                    st.error(message)
                    else:
                        st.error("Please fill in all fields")
                
                if back_button:
                    st.session_state.show_signup = False
                    st.rerun()

def show_main_app(db, model, tokenizer):
    """Display main application interface"""
    # Sidebar
    with st.sidebar:
        st.title(f"👋 Hello, {st.session_state.username}!")
        
        # Navigation
        page = st.selectbox(
            "🧭 Navigate",
            ["💬 Chat", "📄 Documents", "📝 History"],
            index=0
        )
        
        st.divider()
        
        # Model status
        if model and tokenizer:
            st.success("🤖 DexiBot is ready!")
            st.info("💻 Running on CPU")
        else:
            st.error("❌ Model not loaded")
            if st.button("🔄 Reload Model"):
                st.session_state.model = None
                st.session_state.tokenizer = None
                st.rerun()
        
        st.divider()
        
        # Logout button
        if st.button("🚪 Logout", use_container_width=True):
            # Clear session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    # Main content
    if page == "💬 Chat":
        show_chat_interface(db, model, tokenizer)
    elif page == "📄 Documents":
        show_documents_interface(db)
    elif page == "📝 History":
        show_history_interface(db)

def show_chat_interface(db, model, tokenizer):
    """Display chat interface"""
    st.title("💬 Chat with DexiBot")
    
    if not model or not tokenizer:
        st.error("DexiBot model is not loaded. Please check the model files.")
        return
    
    # Chat settings
    with st.expander("⚙️ Chat Settings"):
        col1, col2 = st.columns(2)
        with col1:
            max_length = st.slider("Response Length", 50, 500, 200, 25)
        with col2:
            temperature = st.slider("Creativity", 0.1, 1.0, 0.7, 0.1)
    
    # Load chat history if not already loaded
    if not st.session_state.chat_history:
        st.session_state.chat_history = load_user_chat_history(db, st.session_state.user_id, limit=20)
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for chat in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(chat['message'])
            with st.chat_message("assistant"):
                st.write(chat['response'])
    
    # Chat input
    if prompt := st.chat_input("Ask DexiBot anything..."):
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Generate and display response
        with st.chat_message("assistant"):
            with st.spinner("DexiBot is thinking..."):
                # Retrieve relevant context from user documents
                similar_chunks = retrieve_similar_chunks(db, st.session_state.user_id, prompt, top_k=3)
                
                context = None
                if similar_chunks:
                    # Use only high-similarity chunks (threshold = 0.3)
                    relevant_chunks = [chunk for chunk in similar_chunks if chunk['similarity'] > 0.3]
                    if relevant_chunks:
                        context_parts = [chunk['text'] for chunk in relevant_chunks]
                        context = "\n\n".join(context_parts)
                        # Limit context length
                        if len(context) > 2000:
                            context = context[:2000] + "..."
                
                # Generate response
                response = generate_digibot_response(
                    model, tokenizer, prompt,
                    context=context,
                    max_length=max_length,
                    temperature=temperature
                )
                
                st.write(response)
        
        # Save chat to database and session
        save_chat_message(db, st.session_state.user_id, prompt, response)
        st.session_state.chat_history.append({
            'message': prompt,
            'response': response,
            'timestamp': datetime.now()
        })
        
        st.rerun()
    
    
    # Delete entire conversation from database
    if st.button("Delete Conversation", type="primary"):
        try:
            chats_collection = db[EMPLOYEE_CHATS_COLLECTION]
            result = chats_collection.delete_many({"user_id": st.session_state.user_id})
            st.session_state.chat_history = []
            st.success(f"Deleted {result.deleted_count} chat messages from your history.")
            st.rerun()
        except Exception as e:
            st.error(f"Failed to delete chat history: {e}")

def show_documents_interface(db):
    """Display document management interface"""
    st.title("📄 Document Management")
    
    user_id = st.session_state.user_id
    
    # Upload new document
    st.markdown("### ➕ Upload New Document")
    
    with st.form("upload_form"):
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=["png", "jpg", "jpeg", "pdf", "txt"],
            help="Upload images for OCR extraction or text files"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            doc_title = st.text_input("Document Title*", placeholder="Enter document title")
        with col2:
            doc_description = st.text_input("Description", placeholder="Optional description")
        
        upload_button = st.form_submit_button("📤 Upload & Process", use_container_width=True)
        
        if upload_button:
            if not uploaded_file or not doc_title:
                st.error("Please provide both a file and a title")
            else:
                with st.spinner("Processing document..."):
                    # Save uploaded file temporarily
                    temp_path = f"temp_{uploaded_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.read())
                    
                    try:
                        # Extract text based on file type
                        if uploaded_file.type.startswith('image/'):
                            extracted_text = ocr_extract_text(temp_path)
                        elif uploaded_file.type == 'text/plain':
                            with open(temp_path, 'r', encoding='utf-8') as f:
                                extracted_text = f.read()
                        elif uploaded_file.type == 'application/pdf':
                            try:
                                import PyPDF2
                                with open(temp_path, 'rb') as f:
                                    reader = PyPDF2.PdfReader(f)
                                    extracted_text = "\n".join(page.extract_text() or '' for page in reader.pages)
                            except Exception as e:
                                extracted_text = f"PDF extraction error: {e}"
                        else:
                            extracted_text = "Unsupported file type. Please upload an image, PDF, or text file."
                        
                        if extracted_text and not extracted_text.startswith("OCR Error"):
                            # Save document with embeddings
                            doc_id = save_user_document(db, user_id, doc_title, extracted_text, doc_description)
                            if doc_id:
                                st.success("✅ Document uploaded, processed, and embedded successfully!")
                            else:
                                st.error("Failed to save or embed document.")
                        else:
                            st.error(f"Text extraction failed: {extracted_text}")
                    except Exception as e:
                        st.error(f"Error processing document: {e}")
                    finally:
                        if os.path.exists(temp_path):
                            os.remove(temp_path)

    # List user's uploaded documents
    st.markdown("### 📂 Your Documents")
    user_docs = get_user_documents(db, user_id)
    if user_docs:
        for doc in user_docs:
            doc_key = f"ask_state_{doc['doc_id']}"
            ask_btn_key = f"ask_btn_{doc['doc_id']}"
            with st.expander(f"{doc.get('title', 'Untitled')} (Uploaded: {doc.get('created_at', 'N/A')})"):
                st.write(f"**Description:** {doc.get('description', '')}")
                st.write(f"**Content Preview:**\n{doc.get('content', '')[:500]}{'...' if len(doc.get('content', '')) > 500 else ''}")
                col1, col2 = st.columns(2)
                with col1:
                    # Use session state to persist question and answer for each doc
                    if doc_key not in st.session_state:
                        st.session_state[doc_key] = {'ask': False, 'question': '', 'answer': ''}
                    ask_state = st.session_state[doc_key]
                    if st.button(f"❓ Ask about '{doc.get('title', 'Untitled')}'", key=ask_btn_key):
                        ask_state['ask'] = not ask_state['ask']
                        ask_state['question'] = ''
                        ask_state['answer'] = ''
                    if ask_state['ask']:
                        ask_state['question'] = st.text_input(f"Your question about '{doc.get('title', 'Untitled')}'", value=ask_state['question'], key=f"question_{doc['doc_id']}")
                        if st.button(f"Get Answer for '{doc.get('title', 'Untitled')}'", key=f"get_ans_{doc['doc_id']}"):
                            with st.spinner("Generating answer..."):
                                similar_chunks = retrieve_similar_chunks(db, user_id, ask_state['question'], top_k=3)
                                context = None
                                if similar_chunks:
                                    context_parts = [chunk['text'] for chunk in similar_chunks]
                                    context = "\n\n".join(context_parts)
                                    if len(context) > 2000:
                                        context = context[:2000] + "..."
                                answer = generate_digibot_response(
                                    st.session_state.model, st.session_state.tokenizer,
                                    ask_state['question'], context=context
                                )
                                ask_state['answer'] = answer
                        if ask_state['answer']:
                            st.write(f"**Answer:** {ask_state['answer']}")
                with col2:
                    if st.button(f"🗑️ Delete '{doc.get('title', 'Untitled')}'", key=f"del_{doc['doc_id']}"):
                        ok = delete_user_document(db, user_id, doc['doc_id'])
                        if ok:
                            st.success(f"Deleted document '{doc.get('title', 'Untitled')}' and its embeddings.")
                        else:
                            st.error("Failed to delete document.")
    else:
        st.info("No documents uploaded yet.")


def show_history_interface(db):
    """Display chat history interface"""
    st.title("📝 Chat History")
    full_history = load_user_chat_history(db, st.session_state.user_id, limit=200)
    if not full_history:
        st.info("No chat history found. Start a conversation with DexiBot!")
        return
    col1, col2, col3 = st.columns(3)
    with col1:
        date_filter = st.date_input("Filter by date:", value=None, key="hist_date")
    with col2:
        search_history = st.text_input("Search in history...", key="hist_search")
    with col3:
        limit_results = st.selectbox("Show results:", [10, 25, 50, 100, 200], index=2, key="hist_limit")
    filtered_history = full_history
    if date_filter:
        filtered_history = [
            chat for chat in filtered_history 
            if chat['timestamp'].date() == date_filter
        ]
    if search_history:
        filtered_history = [
            chat for chat in filtered_history 
            if search_history.lower() in chat['message'].lower() or 
               search_history.lower() in chat['response'].lower()
        ]
    filtered_history = filtered_history[-limit_results:]
    st.write(f"Showing {len(filtered_history)} conversations")
    for i, chat in enumerate(reversed(filtered_history), 1):
        ts = chat.get('timestamp')
        ts_str = ts.strftime('%Y-%m-%d %H:%M:%S') if ts else "N/A"
        with st.expander(f"💬 Conversation {i} - {ts_str}"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**You:**")
                st.write(chat['message'])
            with col2:
                st.write("**DexiBot:**")
                st.write(chat['response'])
            st.caption(f"Time: {ts_str}")

# ----------------------------
# Main Application

def main():
    init_session_state()
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .bot-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    </style>
    """, unsafe_allow_html=True)

    db = init_mongodb()
    if db is None:
        st.error("Failed to connect to database. Please check your connection.")
        return

    if st.session_state.model is None or st.session_state.tokenizer is None:
        with st.spinner("Loading DexiBot model..."):
            model, tokenizer = load_digibot_model()
            st.session_state.model = model
            st.session_state.tokenizer = tokenizer

    if not st.session_state.authenticated:
        if st.session_state.show_signup:
            show_signup_form(db)
        else:
            show_login_form(db)
    else:
        show_main_app(db, st.session_state.model, st.session_state.tokenizer)

if __name__ == "__main__":
    main()