from memory.llm import OllamaMemory
import requests
import os

def select_model():
    """Get available models from Ollama"""
    try:
        response = requests.get('http://localhost:11434/api/tags')
        if response.status_code == 200:
            available_models = [model['name'] for model in response.json()['models']]
            print("\nAvailable models:")
            for i, model in enumerate(available_models, 1):
                print(f"{i}. {model}")
            
            while True:
                try:
                    selection = input("\nSelect a model number (or type the model name): ")
                    if selection.isdigit() and 1 <= int(selection) <= len(available_models):
                        return available_models[int(selection)-1]
                    elif selection in available_models:
                        return selection
                    else:
                        print("Invalid selection. Please try again.")
                except ValueError:
                    print("Invalid input. Please enter a number or model name.")
        else:
            print("Could not fetch models. Using default model.")
            return "llama3.1:latest"
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        print("Using default model.")
        return "llama3.1:latest"

if __name__ == "__main__":
    print("Welcome to Memory-Enhanced Chat!")
    model_name = select_model()
    print(f"\nInitializing with model: {model_name}")
    
    memory_system = OllamaMemory(
        model_name=model_name,
        persist_directory="./chroma_db",
        collection_name="conversation_memory"
    )
    
    print("\nChat with me! (Type 'exit' to end)")
    print("Commands:")
    print("  /remember or r - Force memory usage")
    print("  /forget or f - Disable memory for one message")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'exit':
            break
            
        # Check for memory commands with shorter alternatives
        use_memory = True
        if user_input.startswith("/remember") or user_input.startswith("r "):
            user_input = user_input.replace("/remember", "").replace("r ", "").strip()
            print("[Memory explicitly enabled]")
        elif user_input.startswith("/forget") or user_input.startswith("f "):
            user_input = user_input.replace("/forget", "").replace("f ", "").strip()
            use_memory = False
            print("[Memory disabled for this message]")
        
        response = memory_system.chat(user_input, use_memory=use_memory)
        print("\nAssistant:", response)