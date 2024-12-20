import torch
import torch_geometric
import numpy as np
import requests
from datetime import datetime
from torch_geometric.nn import GCNConv
import chromadb
from chromadb.config import Settings
import uuid
import os

class OllamaEmbeddingFunction:
    def __init__(self, model_name):
        self.model_name = model_name
    
    def __call__(self, input):
        if isinstance(input, str):
            input = [input]
            
        embeddings = []
        for text in input:
            response = requests.post(
                'http://localhost:11434/api/embeddings',
                json={'model': self.model_name, 'prompt': text}
            )
            if response.status_code == 200:
                embedding = response.json().get('embedding', [])
                print(f"[DEBUG] Got embedding with dimension: {len(embedding)}")
                embeddings.append(embedding)
            else:
                print(f"[DEBUG] Error getting embedding: {response.status_code}")
                return None
        return embeddings

class MemoryGNN(torch.nn.Module):
    def __init__(self, input_dim=4096, hidden_dim=256, output_dim=128):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

class OllamaMemory:
    def __init__(self, model_name="llama3.1-8bq8-large", persist_directory="./chroma_db"):
        self.model_name = model_name
        self.gnn = MemoryGNN()
        
        # Ensure persistence directory exists
        os.makedirs(persist_directory, exist_ok=True)
        print(f"[DEBUG] Using persistence directory: {os.path.abspath(persist_directory)}")
        
        # Initialize Chroma client with persistence
        self.chroma_client = chromadb.PersistentClient(
            path=persist_directory
        )
        
        # Create embedding function
        self.embedding_function = OllamaEmbeddingFunction(model_name)
        
        # Get or create collection (don't delete existing)
        try:
            self.collection = self.chroma_client.get_collection(
                name="conversation_memory",
                embedding_function=self.embedding_function
            )
            print("[DEBUG] Connected to existing collection")
        except:
            self.collection = self.chroma_client.create_collection(
                name="conversation_memory",
                metadata={"description": "Storage for conversation memory"},
                embedding_function=self.embedding_function
            )
            print("[DEBUG] Created new collection")
        
        # Initialize edge tracking
        self.edge_index = []
        self.last_message_id = None

    def add_message(self, role, content):
        """Add a new message to the memory"""
        # Create unique ID for the message
        message_id = str(uuid.uuid4())
        
        try:
            # Add to Chroma
            self.collection.add(
                documents=[content],
                metadatas=[{
                    'role': role,
                    'timestamp': datetime.now().isoformat(),
                    'message_id': message_id
                }],
                ids=[message_id]
            )
            
            # Update edges for GNN
            if self.last_message_id:
                self.edge_index.append([self.last_message_id, message_id])
                self.edge_index.append([message_id, self.last_message_id])
            
            self.last_message_id = message_id
            print(f"[DEBUG] Successfully stored message in database. Collection size: {self.collection.count()}")
            return True
        except Exception as e:
            print(f"[DEBUG] Error adding message: {e}")
            return False

    def get_relevant_context(self, query, k=10):
        """Get most relevant previous messages for a query"""
        if self.collection.count() == 0:
            return ""
        
        try:
            # Print current query for debugging
            print(f"\n[DEBUG] Searching for context relevant to: {query}")
            
            # Query Chroma for similar messages with higher k value
            results = self.collection.query(
                query_texts=[query],
                n_results=k,
                include=["documents", "metadatas", "distances"]
            )
            
            if not results['documents']:
                return ""
            
            # Format context with similarity scores
            context = "\nRelevant previous conversation:\n"
            for doc, metadata, distance in zip(
                results['documents'][0], 
                results['metadatas'][0],
                results['distances'][0]
            ):
                role = metadata['role']
                similarity = 1 - (distance / 2)  # Convert distance to similarity score
                print(f"[DEBUG] Found memory (similarity: {similarity:.2f}): {doc[:50]}...")
                context += f"{role}: {doc}\n"
            
            return context
        except Exception as e:
            print(f"[DEBUG] Error getting context: {e}")
            return ""

    def chat(self, message, use_memory=True):
        """Chat with memory-enhanced LLM"""
        # Get relevant context from memory if requested
        context = self.get_relevant_context(message) if use_memory else ""
        
        # Log memory usage
        if use_memory and context.strip():
            print("\n[Using memory for this response]")
        
        # Prepare prompt with context
        system_prompt = """You are a helpful AI assistant with access to previous conversation memory. 
When answering questions, always check the provided conversation history carefully and use any relevant information.
If you see information about names (of people, pets, etc.) or personal details in the history, make sure to use them in your response.
If you're not completely sure about something, you can reference the conversation where you learned it.

Here's the relevant conversation history:"""
        
        # Make request to Ollama
        response = requests.post(
            'http://localhost:11434/api/chat',
            json={
                'model': self.model_name,
                'messages': [
                    {"role": "system", "content": system_prompt + context},
                    {"role": "user", "content": message}
                ],
                'stream': False
            }
        )
        
        if response.status_code == 200:
            try:
                assistant_message = response.json()['message']['content']
                
                # Add both user and assistant messages to memory
                if use_memory:
                    self.add_message('user', message)
                    self.add_message('assistant', assistant_message)
                
                return assistant_message
            except Exception as e:
                print(f"[DEBUG] Error parsing response: {e}")
                print(f"[DEBUG] Response content: {response.text}")
                return "Sorry, I encountered an error processing the response."
        else:
            return "Sorry, I encountered an error processing your request."

# Example usage
if __name__ == "__main__":
    # Initialize memory system
    memory_system = OllamaMemory(model_name="llama3.1-8bq8-large")
    
    print("Chat with me! (Type 'exit' to end)")
    print("Commands:")
    print("  /remember - Force memory usage")
    print("  /forget - Disable memory for one message")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'exit':
            break
            
        # Check for memory commands
        use_memory = True
        if user_input.startswith("/remember"):
            user_input = user_input.replace("/remember", "").strip()
            print("[Memory explicitly enabled]")
        elif user_input.startswith("/forget"):
            user_input = user_input.replace("/forget", "").strip()
            use_memory = False
            print("[Memory disabled for this message]")
        
        response = memory_system.chat(user_input, use_memory=use_memory)
        print("\nAssistant:", response)