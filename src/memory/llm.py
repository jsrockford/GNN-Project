import torch
import torch_geometric
import numpy as np
import requests
from datetime import datetime
from torch_geometric.nn import GCNConv
import chromadb
import uuid
import os

class OllamaEmbeddingFunction:
    def __init__(self):
        self.embed_model = "nomic-embed-text"
    
    def __call__(self, input):
        if isinstance(input, str):
            input = [input]
            
        embeddings = []
        for text in input:
            try:
                response = requests.post(
                    'http://localhost:11434/api/embeddings',
                    json={'model': self.embed_model, 'prompt': text}
                )
                if response.status_code == 200:
                    embedding = response.json().get('embedding', [])
                    print(f"[DEBUG] Got embedding with dimension: {len(embedding)}")
                    embeddings.append(embedding)
                else:
                    print(f"[DEBUG] Error getting embedding: {response.status_code}")
                    return None
            except Exception as e:
                print(f"[DEBUG] Embedding error: {e}")
                return None
        return embeddings

class MemoryGNN(torch.nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, output_dim=128):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

class OllamaMemory:
    def __init__(self, model_name="llama3.1:latest", persist_directory="./chroma_db", collection_name="conversation_memory"):
        self.model_name = model_name
        self.gnn = MemoryGNN()
        self.collection_name = collection_name
        
        # Ensure persistence directory exists
        os.makedirs(persist_directory, exist_ok=True)
        print(f"[DEBUG] Using persistence directory: {os.path.abspath(persist_directory)}")
        
        # Initialize Chroma client with persistence
        self.chroma_client = chromadb.PersistentClient(
            path=persist_directory
        )
        
        # Create embedding function
        self.embedding_function = OllamaEmbeddingFunction()
        
        # Create or get collection
        try:
            self.collection = self.chroma_client.get_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
            print("[DEBUG] Connected to existing collection")
        except:
            print("[DEBUG] Creating new collection")
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"description": "Storage for conversation memory"},
                embedding_function=self.embedding_function
            )
        
        # Initialize edge tracking
        self.edge_index = []
        self.last_message_id = None

    def get_relevant_context(self, query, k=15):
        """Get most relevant previous messages for a query"""
        if self.collection.count() == 0:
            return ""
        
        try:
            # Print current query for debugging
            print(f"\n[DEBUG] Searching for context relevant to: {query}")
            
            # Add query variations to improve matching
            queries = [
                query,
                f"{query} details",
                f"information about {query}"
            ]
            
            # Query Chroma for similar messages with multiple queries
            all_results = []
            for q in queries:
                results = self.collection.query(
                    query_texts=[q],
                    n_results=k,
                    include=["documents", "metadatas", "distances"]
                )
                if results['documents']:
                    all_results.extend(zip(
                        results['documents'][0], 
                        results['metadatas'][0],
                        results['distances'][0]
                    ))
            
            # Remove duplicates and sort by relevance
            seen = set()
            unique_results = []
            for doc, metadata, distance in all_results:
                if doc not in seen:
                    seen.add(doc)
                    unique_results.append((doc, metadata, distance))
            
            # Sort by distance and take top k
            unique_results.sort(key=lambda x: x[2])
            unique_results = unique_results[:k]
            
            if not unique_results:
                return ""
            
            # Format context with similarity scores
            context = "\nRelevant previous conversation:\n"
            for doc, metadata, distance in unique_results:
                role = metadata['role']
                similarity = 1 - (distance / 2)  # Convert distance to similarity score
                print(f"[DEBUG] Found memory (similarity: {similarity:.2f}): {doc[:100]}...")
                context += f"{role}: {doc}\n"
            
            return context
        except Exception as e:
            print(f"[DEBUG] Error getting context: {e}")
            return ""

    def add_message(self, role, content):
        """Add a new message to the memory"""
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

    def chat(self, message, use_memory=True):
        """Chat with memory-enhanced LLM"""
        # Get relevant context from memory if requested
        context = self.get_relevant_context(message) if use_memory else ""
        
        # Log memory usage
        if use_memory and context.strip():
            print("\n[Using memory for this response]")
        
        # Prepare prompt with context
        system_prompt = """You are a helpful AI assistant with access to previous conversation memory. 
Follow these rules strictly:
1. Only state facts that are explicitly mentioned in the conversation history
2. If you're not 100% certain about a detail, say "I don't have that information in our conversation history"
3. Never make assumptions or guess about details that weren't explicitly stated
4. If you made a mistake, admit it immediately and correct it
5. Double-check the conversation history before making statements about names, details, or other specific information

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