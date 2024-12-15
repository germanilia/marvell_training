import ollama

# Define our addition tool
def add_two_numbers(a: int, b: int) -> int:
    """
    Add two numbers together

    Args:
        a: The first integer number
        b: The second integer number

    Returns:
        int: The sum of the two numbers
    """
    return a + b

def main():
    print("Welcome to Ollama Calculator CLI!")
    print("Type 'quit' or 'exit' to end the conversation")
    print("Ask me to add numbers...")

    while True:
        # Get user input
        user_input = input("\nYou: ").strip()
        
        # Check for exit command
        if user_input.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break

        print("\nAssistant: Thinking... 🤔")
        
        try:
            # Get response from Ollama
            response = ollama.chat(
                model='llama3.1:8b',
                messages=[{"role": "user", "content": user_input}],
                tools=[add_two_numbers]
            )
            
            # Process the tool calls
            available_functions = {
                'add_two_numbers': add_two_numbers
            }
            
            result = None
            if hasattr(response.message, 'tool_calls') and response.message.tool_calls:
                for tool in response.message.tool_calls:
                    function_to_call = available_functions.get(tool.function.name)
                    if function_to_call:
                        result = function_to_call(**tool.function.arguments)
            
            # Display the result
            if result is not None:
                answer = f"The sum is {result}"
                print(f"Assistant: {answer}")
            else:
                error_msg = "I couldn't calculate that. Please try asking me to add two numbers."
                print(f"Assistant: {error_msg}")
                
        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            print(f"Assistant: {error_msg}")

if __name__ == "__main__":
    main()