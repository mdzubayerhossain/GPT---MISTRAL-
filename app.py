# app.py
from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
import time
import numpy as np
import faiss
from dotenv import load_dotenv
from mistralai import Mistral
from langchain_community.document_loaders import TextLoader
import threading
from itertools import cycle
from cachetools import TTLCache
from queue import Queue
import os
import mysql.connector
import atexit
import uuid
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class ConversationAI:
    def __init__(self, book_path, index_file_path):
        # Database Configuration
        self.db_config = {
            'host': 'localhost',
            'user': 'root',
            'password': '',
            'database': 'user_system',
        }
        
        # API Configuration
        self.apis = [
            {"api_key": "t0zJXzm2HklV9YgNBkjHyCC9z775RYAM", "minute_limit": 500000, "monthly_limit": 100000000, "used_in_last_minute": 0, "used_in_month": 0},
            {"api_key": "AwZDHsbZsOltQX4L3sCBlIKNk5JtsRFE", "minute_limit": 500000, "monthly_limit": 100000000, "used_in_last_minute": 0, "used_in_month": 0},
            {"api_key": "wtiP9b9HFvZdbyhTByQLjLeAXctbzp3F", "minute_limit": 500000, "monthly_limit": 100000000, "used_in_last_minute": 0, "used_in_month": 0},
        ]
        
        # Configuration Constants
        self.MAX_MEMORY_LENGTH = 10
        self.MAX_QUESTION_FREQUENCY = 3
        self.MAX_SESSION_DURATION = 3600  # 1 hour
        self.DELAY_SECONDS = 2
        
        # Initialize resources
        self.api_cycle = cycle(self.apis)
        self.request_queue = Queue()
        self.cache = TTLCache(maxsize=100, ttl=300)
        
        # Database Connection
        self.conn = mysql.connector.connect(**self.db_config)
        self.cursor = self.conn.cursor(dictionary=True)
        
        # Session Management
        self.user_sessions = {}
        
        # Text Processing
        self.book_path = book_path
        self.index_file_path = index_file_path
        self.load_text_data()
        
        # Start background threads
        self.start_background_tasks()
    
    def load_text_data(self):
        # Load and chunk text data
        loader = TextLoader(self.book_path, encoding="utf-8")
        docs = loader.load()
        text = docs[0].page_content
        
        # Chunk text
        self.chunk_size = 6500
        self.chunks = [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]
        
        # Setup Faiss Index
        self.setup_faiss_index()
    
    def setup_faiss_index(self):
        if os.path.exists(self.index_file_path):
            self.index = faiss.read_index(self.index_file_path)
            logger.info("Loaded existing Faiss index.")
        else:
            text_embeddings = []
            for chunk in self.chunks:
                embedding = self.get_text_embedding(chunk)
                text_embeddings.append(embedding)
                time.sleep(self.DELAY_SECONDS)
            
            text_embeddings = np.array(text_embeddings)
            d = text_embeddings.shape[1]
            self.index = faiss.IndexFlatL2(d)
            self.index.add(text_embeddings)
            faiss.write_index(self.index, self.index_file_path)
            logger.info("New Faiss index created and saved.")
    
    def get_next_api(self):
        for api in self.api_cycle:
            if (api["used_in_last_minute"] < api["minute_limit"] and 
                api["used_in_month"] < api["monthly_limit"]):
                api["used_in_last_minute"] += 1
                api["used_in_month"] += 1
                return Mistral(api_key=api["api_key"])
        raise Exception("All APIs have exceeded their limits.")
    
    def get_text_embedding(self, input_text):
        try:
            client = self.get_next_api()
            embeddings_batch_response = client.embeddings.create(
                model="mistral-embed",
                inputs=[input_text]
            )
            return embeddings_batch_response.data[0].embedding
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            raise
    
    def start_background_tasks(self):
        # Reset usage statistics threads
        threading.Thread(target=self.reset_minute_usage, daemon=True).start()
        threading.Thread(target=self.reset_monthly_usage, daemon=True).start()
        
        # Request processing thread
        threading.Thread(target=self.process_requests, daemon=True).start()
    
    def reset_minute_usage(self):
        while True:
            time.sleep(60)
            for api in self.apis:
                api["used_in_last_minute"] = 0
    
    def reset_monthly_usage(self):
        while True:
            time.sleep(30 * 24 * 60 * 60)
            for api in self.apis:
                api["used_in_month"] = 0
    
    def create_session(self):
        session_id = str(uuid.uuid4())
        self.user_sessions[session_id] = {
            "conversation_memory": [],
            "question_counts": {},
            "last_active": time.time()
        }
        return session_id
    
    def manage_conversation_memory(self, session_id):
        session = self.user_sessions.get(session_id)
        if session and len(session['conversation_memory']) > self.MAX_MEMORY_LENGTH * 2:
            session['conversation_memory'] = session['conversation_memory'][-self.MAX_MEMORY_LENGTH * 2:]
    
    def generate_conversation_history(self, session_id):
        session = self.user_sessions.get(session_id, {})
        memory = session.get('conversation_memory', [])
        
        history = ""
        for entry in memory:
            history += f"{entry['role'].capitalize()}: {entry['content']}\n"
        return history
    
    def run_mistral(self, prompt, model="open-mistral-nemo"):
        client = self.get_next_api()
        messages = [{"role": "user", "content": prompt}]
        time.sleep(self.DELAY_SECONDS)
        chat_response = client.chat.complete(model=model, messages=messages)
        return chat_response.choices[0].message.content
    
    def check_in_database(self, question):
        query = "SELECT response FROM conversation_cache WHERE question = %s"
        self.cursor.execute(query, (question,))
        result = self.cursor.fetchone()
        return result['response'] if result else None
    
    def save_to_database(self, question, response):
        query = "INSERT INTO conversation_cache (question, response) VALUES (%s, %s)"
        self.cursor.execute(query, (question, response))
        self.conn.commit()
    
    def process_requests(self):
        while True:
            question, session_id, callback = self.request_queue.get()
            try:
                session = self.user_sessions.get(session_id)
                if not session:
                    session = self.create_session()
                
                # Track repeated questions
                session['question_counts'][question] = session['question_counts'].get(question, 0) + 1
                
                if session['question_counts'][question] >= self.MAX_QUESTION_FREQUENCY:
                    session['question_counts'][question] = 0
                    answer = self.check_in_database(question) or "No response found."
                else:
                    # Check cache and database
                    cached_response = self.check_in_database(question)
                    if cached_response:
                        answer = cached_response
                    else:
                        # Semantic search and response generation
                        question_embedding = np.array([self.get_text_embedding(question)])
                        D, I = self.index.search(question_embedding, k=2)
                        retrieved_chunk = [self.chunks[i] for i in I[0]]
                        
                        prompt = f"""
                        Context: {retrieved_chunk}
                        Previous Conversation:
                        {self.generate_conversation_history(session_id)}
                        
                        Query: {question}
                        Answer in Bengali, based on the context and previous conversation:
                        """
                        
                        answer = self.run_mistral(prompt)
                        self.save_to_database(question, answer)
                
                # Update conversation memory
                session['conversation_memory'].append({
                    "role": "user",
                    "content": question,
                    "timestamp": time.time()
                })
                session['conversation_memory'].append({
                    "role": "assistant",
                    "content": answer,
                    "timestamp": time.time()
                })
                
                # Manage conversation memory
                self.manage_conversation_memory(session_id)
                
                callback(answer, session_id)
            
            except Exception as e:
                logger.error(f"Error processing request: {e}")
                callback(str(e), session_id)
            finally:
                self.request_queue.task_done()
    
    def process_request(self, question, session_id=None):
        if not session_id:
            session_id = self.create_session()
        
        response_holder = []
        
        def callback(response, sid):
            response_holder.append((response, sid))
        
        self.request_queue.put((question, session_id, callback))
        self.request_queue.join()
        
        return response_holder[0]

# Flask Application
def create_app(conversation_ai):
    app = Flask(__name__)
    CORS(app)
    app.secret_key = os.urandom(24)
    
    @app.route('/')
    def home():
        return render_template('index.html')
    
    @app.route('/get', methods=['GET'])
    def get_bot_response():
        question = request.args.get('msg')
        session_id = request.args.get('session_id')
        
        response, new_session_id = conversation_ai.process_request(question, session_id)
        
        return jsonify({
            'response': response,
            'session_id': new_session_id
        })
    
    @atexit.register
    def close_connection():
        if conversation_ai.conn.is_connected():
            conversation_ai.cursor.close()
            conversation_ai.conn.close()
    
    return app

# Main Execution
if __name__ == "__main__":
    conversation_ai = ConversationAI(
        book_path=r"D:\Coding\reqq\book.txt", 
        index_file_path=r"embeddings_index_new.faiss"
    )
    
    app = create_app(conversation_ai)
    app.run(debug=True)