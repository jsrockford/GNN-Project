import time
import requests
from memory_llm import OllamaMemory

def select_model():
    """Get available models from Ollama and let user select one"""
    try:
        response = requests.get('http://localhost:11434/api/tags')
        if response.status_code == 200:
            available_models = [model['name'] for model in response.json()['models']]
            print("\nAvailable models:")
            for i, model in enumerate(available_models, 1):
                print(f"{i}. {model}")
            
            # Get user selection
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

class MemoryTester:
    def __init__(self, model_name, test_db_path="./chroma_test_db"):
        """Initialize tester with separate test database"""
        self.model_name = model_name
        self.test_db_path = test_db_path
        # Initialize memory system with test database
        self.memory = OllamaMemory(model_name=model_name, persist_directory=test_db_path)
        self.test_results = []
        print(f"[DEBUG] Initialized test database at: {os.path.abspath(test_db_path)}")
    
    def clear_database(self):
        """Clear the test database"""
        try:
            self.memory.chroma_client.delete_collection("conversation_memory")
            print("\n[DEBUG] Cleared test collection")
            # Recreate empty collection
            self.memory.collection = self.memory.chroma_client.create_collection(
                name="conversation_memory",
                metadata={"description": "Test conversation memory"},
                embedding_function=self.memory.embedding_function
            )
        except Exception as e:
            print(f"\n[DEBUG] Error clearing database: {e}")
    
    def run_test(self, test_name, inputs, queries, expected_results, clear_first=False):
        """Run a single test case"""
        if clear_first:
            self.clear_database()
            
        print(f"\n=== Running Test: {test_name} ===")        
        # Input phase - providing information
        print("\nProviding information...")
        memory_system1 = OllamaMemory(model_name=self.memory.model_name)
        for input_text in inputs:
            print(f"\nInput: {input_text}")
            response = memory_system1.chat(f"r {input_text}", use_memory=True)
            print(f"Response: {response}")
        
        print("\nClosing initial conversation...")
        del memory_system1  # Close first conversation
        
        # Optional delay to simulate time passing
        time.sleep(1)
        
        # Query phase - testing recall with fresh conversation
        print("\nStarting new conversation for recall testing...")
        memory_system2 = OllamaMemory(model_name=self.memory.model_name)
        
        results = []
        print("\nTesting recall...")
        for query, expected in zip(queries, expected_results):
            print(f"\nQuery: {query}")
            response = memory_system2.chat(f"r {query}", use_memory=True)
            print(f"Response: {response}")
            
            # Simple check if expected information is in response
            passed = any(expect.lower() in response.lower() for expect in expected)
            results.append({
                'query': query,
                'response': response,
                'expected': expected,
                'passed': passed
            })
        
        # Clean up
        del memory_system2
        
        # Store test results
        self.test_results.append({
            'test_name': test_name,
            'results': results
        })
        
        return results

    def print_results(self):
        """Print summary of all test results"""
        print("\n=== Test Results Summary ===")
        for test in self.test_results:
            passed = sum(1 for r in test['results'] if r['passed'])
            total = len(test['results'])
            print(f"\n{test['test_name']}: {passed}/{total} passed")
            
            for result in test['results']:
                status = "✓" if result['passed'] else "✗"
                print(f"\n{status} Query: {result['query']}")
                print(f"  Expected to contain: {result['expected']}")
                print(f"  Response: {result['response']}")

def run_memory_tests(memory_system):
    # First run error handling tests with fresh database
    print("\n=== Running Error Handling Tests First (Fresh Database) ===")
    error_tester = MemoryTester(memory_system)
    error_tester.run_test(
        "Error Handling Test",
        inputs=["I have a dog named Buddy"],
        queries=[
            "What's my phone number?",  # Something never mentioned
            "What's my cat's name?",    # No cat was mentioned
            "What color is my car?"     # No car was mentioned
        ],
        expected_results=[
            ["don't have", "not mentioned", "no information"],
            ["don't have", "not mentioned", "no information"],
            ["don't have", "not mentioned", "no information"]
        ],
        clear_first=True  # Clear database before this test
    )
    
    print("\n=== Running Regular Memory Tests ===")
    tester = MemoryTester(memory_system)
    
    # Test 1: Simple Facts
    tester.run_test(
        "Simple Facts Test",
        inputs=["My name is Alice", "I am 30 years old"],
        queries=["What's my name?", "How old am I?"],
        expected_results=[["Alice"], ["30"]]
    )
    
    # Test 2: Compound Facts
    tester.run_test(
        "Compound Facts Test",
        inputs=["I have a cat named Max who loves tuna"],
        queries=[
            "What's my cat's name?",
            "What does my cat like?",
            "Do I have any pets?"
        ],
        expected_results=[
            ["Max"],
            ["tuna"],
            ["cat", "Max"]
        ]
    )
    
    # Test 3: Sequential Information
    tester.run_test(
        "Sequential Information Test",
        inputs=[
            "I went to Paris last summer",
            "While in Paris, I visited the Eiffel Tower",
            "After Paris, I went to Rome"
        ],
        queries=[
            "Where did I go last summer?",
            "What did I visit in Paris?",
            "Where did I go after Paris?"
        ],
        expected_results=[
            ["Paris"],
            ["Eiffel Tower"],
            ["Rome"]
        ]
    )
    
    # Test 4: Implicit Information
    tester.run_test(
        "Implicit Information Test",
        inputs=[
            "The golden retriever bounded across the yard",
            "Sophie loves playing fetch with her tennis ball"
        ],
        queries=[
            "What kind of dog is Sophie?",
            "What does Sophie like to play with?"
        ],
        expected_results=[
            ["golden retriever"],
            ["tennis ball"]
        ]
    )
    
    # Test 5: Multiple Subjects
    tester.run_test(
        "Multiple Subjects Test",
        inputs=[
            "Tom is a software engineer who codes in Python",
            "Sarah is a doctor who works in pediatrics",
            "Mike is a chef who specializes in Italian cuisine"
        ],
        queries=[
            "What does Tom do?",
            "What kind of doctor is Sarah?",
            "What kind of food does Mike cook?"
        ],
        expected_results=[
            ["software engineer", "Python"],
            ["pediatrics"],
            ["Italian"]
        ]
    )
    
    # Test 6: Error Handling
    tester.run_test(
        "Error Handling Test",
        inputs=["I have a dog named Buddy"],
        queries=[
            "What's my cat's name?",  # Should indicate no cat was mentioned
            "What's my dog's favorite food?"  # Should indicate this wasn't mentioned
        ],
        expected_results=[
            ["don't have", "not mentioned", "no information"],
            ["don't have", "not mentioned", "no information"]
        ]
    )
    
    # Print results
    tester.print_results()

if __name__ == "__main__":
    print("Welcome to Memory System Testing!")
    print("First, let's select a model to test with.")
    
    # Get model selection from user
    model_name = select_model()
    print(f"\nInitializing with model: {model_name}")
    
    # Create test directory if it doesn't exist
    test_db_path = "./chroma_test_db"
    os.makedirs(test_db_path, exist_ok=True)
    
    print("\nNote: Tests will use a separate database and won't affect your existing conversations.")
    print(f"Test database location: {os.path.abspath(test_db_path)}")
    
    # Initialize tester with selected model
    error_tester = MemoryTester(model_name, test_db_path)
    
    # Run error handling tests first
    print("\n=== Running Error Handling Tests First (Fresh Database) ===")
    error_tester.run_test(
        "Error Handling Test",
        inputs=["I have a dog named Buddy"],
        queries=[
            "What's my phone number?",
            "What's my cat's name?",
            "What color is my car?"
        ],
        expected_results=[
            ["don't have", "not mentioned", "no information"],
            ["don't have", "not mentioned", "no information"],
            ["don't have", "not mentioned", "no information"]
        ],
        clear_first=True
    )
    
    # Run regular tests
    print("\n=== Running Regular Memory Tests ===")
    tester = MemoryTester(model_name, test_db_path)
    
    # Run remaining tests...
    tester.run_test(
        "Simple Facts Test",
        inputs=["My name is Alice", "I am 30 years old"],
        queries=["What's my name?", "How old am I?"],
        expected_results=[["Alice"], ["30"]]
    )