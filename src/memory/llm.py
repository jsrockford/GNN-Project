import torch
import torch_geometric
import numpy as np
import requests
from datetime import datetime
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import chromadb
import uuid
import os
import json
import pickle

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
    def __init__(self, input_dim=768, hidden_dim=1024, output_dim=768):  # Maintaining 768 dimensions
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        
    def forward(self, x, edge_index, edge_weights=None):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

class OllamaMemory:
    def __init__(self, model_name="llama3.1:latest", persist_directory="./chroma_db", collection_name="conversation_memory"):
        self.model_name = model_name
        self.persist_directory = persist_directory
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gnn = MemoryGNN().to(self.device)
        self.collection_name = collection_name
        
        # Graph components storage
        self.node_embeddings = []
        self.node_ids = []
        self.edge_index = []
        self.edge_weights = []
        self.graph_file = os.path.join(persist_directory, "graph_data.pkl")
        
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
            
            # Load existing graph data if available
            self.load_graph_data()
            
        except:
            print("[DEBUG] Creating new collection")
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"description": "Storage for conversation memory"},
                embedding_function=self.embedding_function
            )

    def save_graph_data(self):
        """Save graph components to disk"""
        graph_data = {
            'node_embeddings': self.node_embeddings,
            'node_ids': self.node_ids,
            'edge_index': self.edge_index,
            'edge_weights': self.edge_weights
        }
        try:
            with open(self.graph_file, 'wb') as f:
                pickle.dump(graph_data, f)
            print("[DEBUG] Graph data saved successfully")
        except Exception as e:
            print(f"[DEBUG] Error saving graph data: {e}")

    def load_graph_data(self):
        """Load graph components from disk"""
        try:
            if os.path.exists(self.graph_file):
                with open(self.graph_file, 'rb') as f:
                    graph_data = pickle.load(f)
                self.node_embeddings = graph_data['node_embeddings']
                self.node_ids = graph_data['node_ids']
                self.edge_index = graph_data['edge_index']
                self.edge_weights = graph_data['edge_weights']
                print("[DEBUG] Graph data loaded successfully")
            else:
                print("[DEBUG] No existing graph data found")
                # Rebuild graph from ChromaDB if needed
                self.rebuild_graph_from_chroma()
        except Exception as e:
            print(f"[DEBUG] Error loading graph data: {e}")
            self.rebuild_graph_from_chroma()

    def rebuild_graph_from_chroma(self):
        """Rebuild graph structure from ChromaDB data"""
        try:
            # Get all documents from ChromaDB
            results = self.collection.get(
                include=['documents', 'embeddings', 'metadatas']
            )
            
            # Reset graph components
            self.node_embeddings = []
            self.node_ids = []
            self.edge_index = []
            self.edge_weights = []
            
            # Rebuild nodes
            for i, (doc, embedding, metadata) in enumerate(zip(
                results['documents'],
                results['embeddings'],
                results['metadatas']
            )):
                self.node_embeddings.append(embedding)
                self.node_ids.append(metadata['message_id'])
                
                # Add edges to previous message
                if i > 0:
                    prev_idx = i - 1
                    self.edge_index.extend([
                        [prev_idx, i],
                        [i, prev_idx]
                    ])
                    # Calculate similarity for edge weight
                    prev_emb = torch.tensor(self.node_embeddings[prev_idx])
                    curr_emb = torch.tensor(embedding)
                    similarity = torch.cosine_similarity(prev_emb, curr_emb, dim=0)
                    self.edge_weights.extend([similarity.item(), similarity.item()])
            
            print(f"[DEBUG] Rebuilt graph with {len(self.node_ids)} nodes")
            self.save_graph_data()
            
        except Exception as e:
            print(f"[DEBUG] Error rebuilding graph: {e}")

    def build_graph(self):
        """Construct a PyG graph from stored components"""
        if not self.node_embeddings:
            return None
            
        try:
            # Convert lists to tensors
            x = torch.tensor(self.node_embeddings, dtype=torch.float32).to(self.device)
            edge_index = torch.tensor(self.edge_index, dtype=torch.long).t().contiguous().to(self.device)
            edge_weights = torch.tensor(self.edge_weights, dtype=torch.float32).to(self.device)
            
            # Create PyG Data object
            graph = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_weights
            )
            
            return graph
            
        except Exception as e:
            print(f"[DEBUG] Error building graph: {e}")
            return None

    def get_relevant_context(self, query, k=15):
        """Get most relevant previous messages using GNN-enhanced embeddings"""
        if self.collection.count() == 0:
            return ""
        
        try:
            print("\n[DEBUG] Attempting GNN-based memory retrieval...")
            # Get query embedding
            query_embedding = self.embedding_function([query])[0]
            
            # Build and process graph
            graph = self.build_graph()
            if graph is not None:
                # Process embeddings through GNN
                with torch.no_grad():
                    enhanced_embeddings = self.gnn(
                        graph.x,
                        graph.edge_index,
                        graph.edge_attr
                    )
                
                # Convert query embedding to tensor
                query_tensor = torch.tensor(query_embedding, dtype=torch.float32).to(self.device)
                
                # Calculate similarities with enhanced embeddings
                similarities = torch.cosine_similarity(
                    query_tensor.unsqueeze(0),
                    enhanced_embeddings
                )
                
                # Get top k similar indices
                top_k_indices = similarities.argsort(descending=True)[:k].cpu().numpy()
                
                # Retrieve messages using node_ids
                context = "\n[Using GNN-Enhanced Memory Retrieval]\nRelevant previous conversation:\n"
                memories_found = False
                
                for idx in top_k_indices:
                    message_id = self.node_ids[idx]
                    # Get message from ChromaDB
                    result = self.collection.get(
                        ids=[message_id],
                        include=['documents', 'metadatas']
                    )
                    if result['documents']:
                        memories_found = True
                        role = result['metadatas'][0]['role']
                        content = result['documents'][0]
                        similarity_score = similarities[idx].item()
                        print(f"[DEBUG] GNN Memory (similarity: {similarity_score:.2f}): {content[:100]}...")
                        context += f"{role}: {content}\n"
                
                if memories_found:
                    return context
            
            # Fallback to regular ChromaDB search
            print("[DEBUG] Falling back to ChromaDB retrieval...")
            results = self.collection.query(
                query_texts=[query],
                n_results=k,
                include=["documents", "metadatas", "distances"]
            )
            
            if not results['documents']:
                return ""
            
            context = "\n[Using ChromaDB Retrieval - GNN Fallback]\nRelevant previous conversation:\n"
            for doc, metadata, distance in zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            ):
                role = metadata['role']
                similarity = 1 - (distance / 2)
                print(f"[DEBUG] ChromaDB Memory (similarity: {similarity:.2f}): {doc[:100]}...")
                context += f"{role}: {doc}\n"
            
            return context
            
        except Exception as e:
            print(f"[DEBUG] Error getting context: {e}")
            return ""

    def add_message(self, role, content):
        """Add a new message to the memory"""
        message_id = str(uuid.uuid4())
        
        try:
            # Get embedding for new message
            embedding = self.embedding_function([content])[0]
            
            # Add to ChromaDB
            self.collection.add(
                documents=[content],
                metadatas=[{
                    'role': role,
                    'timestamp': datetime.now().isoformat(),
                    'message_id': message_id
                }],
                ids=[message_id]
            )
            
            # Update graph components
            self.node_embeddings.append(embedding)
            self.node_ids.append(message_id)
            
            # Add edges to previous message
            if len(self.node_ids) > 1:
                prev_idx = len(self.node_ids) - 2
                curr_idx = len(self.node_ids) - 1
                self.edge_index.extend([
                    [prev_idx, curr_idx],
                    [curr_idx, prev_idx]  # Bidirectional
                ])
                
                # Calculate similarity for edge weight
                prev_emb = torch.tensor(self.node_embeddings[prev_idx])
                curr_emb = torch.tensor(embedding)
                similarity = torch.cosine_similarity(prev_emb, curr_emb, dim=0)
                self.edge_weights.extend([similarity.item(), similarity.item()])
            
            # Save updated graph data
            self.save_graph_data()
            
            print(f"[DEBUG] Successfully stored message and updated graph. Nodes: {len(self.node_ids)}")
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