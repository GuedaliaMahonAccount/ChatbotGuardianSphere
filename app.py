import os
import uuid
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from flask_cors import CORS
from pymongo import MongoClient
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MONGODB_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")

# Initialize Flask
app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": ['https://guardian-sphere.azurewebsites.net', 'http://localhost:3000',
                    'https://guardianspheres.com'],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True
    }
})

# MongoDB setup
client = MongoClient(MONGODB_URI)
db = client.get_database("chat_db")
chat_collection = db.get_collection("chats")

# Global variables to store PDF data
vector_store = None
pdf_loaded = False


def load_pdfs_at_startup():
    """Load all PDFs from a specified directory at startup"""
    global vector_store, pdf_loaded

    pdf_directory = "pdf"  # Change this to your PDF directory
    combined_text = ""

    try:
        for filename in os.listdir(pdf_directory):
            if filename.endswith('.pdf'):
                pdf_path = os.path.join(pdf_directory, filename)
                pdf_reader = PdfReader(pdf_path)

                for page in pdf_reader.pages:
                    text = page.extract_text()
                    if text:
                        combined_text += text + "\n"

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", ".", "!", "?", " ", ""],
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len
        )
        chunks = text_splitter.split_text(combined_text)

        # Create embeddings and vector store
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        vector_store = FAISS.from_texts(chunks, embeddings)
        pdf_loaded = True

        print("PDFs loaded successfully")
        return True
    except Exception as e:
        print(f"Error loading PDFs: {str(e)}")
        return False


def get_ai_response(query):
    """Get response from AI using loaded PDF knowledge"""
    if not pdf_loaded or vector_store is None:
        return "PDF data not loaded properly"

    try:
        # Find similar chunks
        results = vector_store.similarity_search(query, k=2)

        # Initialize ChatOpenAI
        llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            temperature=0,
            max_tokens=100,
            model_name="gpt-4"
        )

        # Create and run QA chain
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=results, question=query)

        return response
    except Exception as e:
        return f"Error generating response: {str(e)}"


# Your existing routes with modified chat functionality
@app.route("/")
def home():
    return jsonify({"message": "Welcome to the Azure OpenAI GPT-powered chat application"})

@app.route("/new-chat", methods=["POST", "OPTIONS"])
def new_chat():
    if request.method == "OPTIONS":
        return {}, 200

    try:
        username = request.json.get("username")
        title = request.json.get("title")

        if not username or not title:
            return jsonify({"error": "Username and title are required"}), 400

        chat_id = str(uuid.uuid4())

        chat_data = {
            "_id": chat_id,
            "username": username,
            "title": title,
            "messages": [],
            "feedback": None  # Add this line
        }

        chat_collection.insert_one(chat_data)

        return jsonify({
            "chat": {
                "_id": chat_id,
                "title": title,
                "messages": [],
                "feedback": None
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/chat", methods=["POST", "OPTIONS"])
def chat():
    if request.method == "OPTIONS":
        return {}, 200

    try:
        username = request.json.get("username")
        chat_id = request.json.get("chatId")
        user_message = request.json.get("message", "").strip()

        if not all([username, chat_id, user_message]):
            return jsonify({"error": "Missing required fields"}), 400

        # Get AI response using PDF knowledge
        ai_response = get_ai_response(user_message)

        # Update chat history
        chat_messages = chat_collection.find_one({"_id": chat_id})["messages"]
        chat_messages.extend([
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": ai_response}
        ])

        # Save to database
        chat_collection.update_one(
            {"_id": chat_id, "username": username},
            {"$set": {"messages": chat_messages}}
        )

        return jsonify({
            "response": ai_response,
            "messages": chat_messages
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/history/<username>", methods=["GET", "OPTIONS"])
def get_chat_history(username):
    if request.method == "OPTIONS":
        return {}, 200

    try:
        chats = list(chat_collection.find({"username": username}))
        history = [
            {
                "_id": chat["_id"],
                "title": chat["title"],
                "messages": chat["messages"]
            }
            for chat in chats
        ]
        return jsonify({"history": history})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/delete-chat", methods=["DELETE", "OPTIONS"])
def delete_chat():
    if request.method == "OPTIONS":
        return {}, 200

    try:
        username = request.json.get("username")
        chat_id = request.json.get("chatId")

        if not username or not chat_id:
            return jsonify({"error": "Username and chatId are required"}), 400

        result = chat_collection.delete_one({"_id": chat_id, "username": username})
        if result.deleted_count == 0:
            return jsonify({"error": "Chat not found"}), 404

        return jsonify({"message": "Chat deleted successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/update-chat-title", methods=["PUT", "OPTIONS"])
def update_chat_title():
    if request.method == "OPTIONS":
        return {}, 200

    try:
        username = request.json.get("username")
        chat_id = request.json.get("chatId")
        new_title = request.json.get("newTitle")

        if not username or not chat_id or not new_title:
            return jsonify({"error": "Username, chatId, and newTitle are required"}), 400

        result = chat_collection.update_one(
            {"_id": chat_id, "username": username},
            {"$set": {"title": new_title}}
        )

        if result.matched_count == 0:
            return jsonify({"error": "Chat not found"}), 404

        return jsonify({"message": "Chat title updated successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/update-feedback", methods=["PUT", "OPTIONS"])
def update_chat_feedback():
    if request.method == "OPTIONS":
        return {}, 200

    try:
        # Log the incoming request
        print("Incoming feedback update request...")

        # Extract and log input data
        username = request.json.get("username")
        chat_id = request.json.get("chatId")
        feedback = request.json.get("feedback")
        print("Received data:", {"username": username, "chatId": chat_id, "feedback": feedback})

        # Validate fields and feedback value
        if not username or not chat_id or feedback not in ["like", "dislike"]:
            print("Validation failed: missing fields or invalid feedback value.")
            return jsonify({"error": "Invalid feedback or missing fields."}), 400

        # Perform MongoDB query and log result
        document = chat_collection.find_one({"_id": chat_id, "username": username})
        print("Document found in MongoDB:", document)

        if not document:
            print("No matching document found for username and chatId.")
            return jsonify({"error": "Chat not found"}), 404

        # Update feedback in MongoDB
        result = chat_collection.update_one(
            {"_id": chat_id, "username": username},
            {"$set": {"feedback": feedback}}
        )
        print("Update result:", result.raw_result)

        if result.matched_count == 0:
            print("No document matched for update.")
            return jsonify({"error": "Chat not found"}), 404

        print("Feedback updated successfully.")
        return jsonify({"message": "Feedback updated successfully"})
    except Exception as e:
        print("Error in update_chat_feedback:", str(e))
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Load PDFs at startup
    if load_pdfs_at_startup():
        port = int(os.environ.get("PORT", 8000))
        app.run(host="0.0.0.0", port=port)
    else:
        print("Failed to load PDFs. Server not started.")