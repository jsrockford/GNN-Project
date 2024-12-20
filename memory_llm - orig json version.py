import torch
import torch_geometric
import numpy as np
import json
import requests
from datetime import datetime
from torch_geometric.nn import GCNConv
import os

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
    def __init__(self, model_name="llama3.1:latest", memory_file="conversation_memory.json"):
        self.model_name = model_name
        self.memory_file = memory_file
        self.gnn = MemoryGNN()
        self.messages = []
        self.embeddings = []
        self.edge_index = []
        
        # Load existing memory if it exists
        self.load_memory()
    
    def get_embedding(self, text):
        """Get embedding from Ollama"""
        response = requests.post(
            'http://localhost:11434/api/embeddings',
            json={'model': self.model_name, 'prompt': text}
        )
        if response.status_code == 200:
            embedding = response.json().get('embedding', [])
            return torch.tensor(embedding)
        return None

    def save_memory(self):
        """Save conversations and graph structure to file"""
        memory_data = {
            'messages': self.messages,
            'embeddings': [e.tolist() for e in self.embeddings],
            'edge_index': self.edge_index
        }
        with open(self.memory_file, 'w') as f:
            json.dump(memory_data, f)

    def load_memory(self):
        """Load conversations and graph structure from file"""
        if os.path.exists(self.memory_file):
            with open(self.memory_file, 'r') as f:
                memory_data = json.load(f)
                self.messages = memory_data['messages']
                self.embeddings = [torch.tensor(e) for e in memory_data['embeddings']]
                self.edge_index = memory_data['edge_index']

    def add_message(self, role, content):
        """Add a new message to the memory"""
        # Create message object
        message = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat()
        }
        
        # Get embedding
        embedding = self.get_embedding(content)
        if embedding is None:
            return False
        
        # Add to memory
        msg_idx = len(self.messages)
        self.messages.append(message)
        self.embeddings.append(embedding)
        
        # Add edges to previous messages
        if msg_idx > 0:
            self.edge_index.append([msg_idx-1, msg_idx])
            self.edge_index.append([msg_idx, msg_idx-1])  # Bidirectional
        
        # Save to file
        self.save_memory()
        return True

    def get_relevant_context(self, query, k=5):
        """Get most relevant previous messages for a query"""
        if not self.messages:
            return ""
        
        # Get query embedding
        query_embedding = self.get_embedding(query)
        if query_embedding is None:
            return ""
        
        # Calculate similarities
        similarities = []
        for emb in self.embeddings:
            sim = torch.cosine_similarity(query_embedding, emb, dim=0)
            similarities.append(sim.item())
        
        # Get top k most similar messages
        top_k_idx = np.argsort(similarities)[-k:][::-1]
        
        # Format context
        context = "\nRelevant previous conversation:\n"
        for idx in top_k_idx:
            msg = self.messages[idx]
            context += f"{msg['role']}: {msg['content']}\n"
        
        return context

    def chat(self, message, use_memory=True):
        """Chat with memory-enhanced LLM
        Args:
            message (str): The user's message
            use_memory (bool): Whether to use memory retrieval
        """
        # Get relevant context from memory only if requested
        context = self.get_relevant_context(message) if use_memory else ""
        
        # Log whether memory was used
        if use_memory:
            print("\n[Using memory for this response]")
        
        # Prepare prompt with context
        system_prompt = "You are a helpful AI assistant with access to previous conversation memory. Use the relevant context provided to inform your responses, maintaining consistency with previous interactions."
        
        # Make request to Ollama
        response = requests.post(
            'http://localhost:11434/api/chat',
            json={
                'model': self.model_name,
                'messages': [
                    {"role": "system", "content": system_prompt + context},
                    {"role": "user", "content": message}
                ],
                'stream': False  # Request complete response instead of stream
            }
        )
        
        if response.status_code == 200:
            try:
                assistant_message = response.json()['message']['content']
                
                # Add both user and assistant messages to memory
                self.add_message('user', message)
                self.add_message('assistant', assistant_message)
                
                return assistant_message
            except Exception as e:
                print(f"Error parsing response: {e}")
                print(f"Response content: {response.text}")
                return "Sorry, I encountered an error processing the response."
        else:
            return "Sorry, I encountered an error processing your request."

# Example usage
if __name__ == "__main__":
    # Initialize memory system
    memory_system = OllamaMemory(model_name="llama3.1:latest")
    
    # Chat loop
    print("Chat with me! (Type 'exit' to end)")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'exit':
            break
            
        response = memory_system.chat(user_input)
        print("\nAssistant:", response)