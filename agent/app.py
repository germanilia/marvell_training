import os
import ollama
import http.client
import json
from bs4 import BeautifulSoup # type: ignore
def search_serper(query: str) -> str:
    """
    Search the web using Serper API

    Args:
        query: The search query string

    Returns:
        str: The search results as a formatted string
    """
    try:
        conn = http.client.HTTPSConnection("google.serper.dev")
        payload = json.dumps({
            "q": query
        })
        headers = {
            'X-API-KEY': os.getenv("SERPER_API_KEY"),
            'Content-Type': 'application/json'
        }
        
        conn.request("POST", "/search", payload, headers)
        res = conn.getresponse()
        data = json.loads(res.read().decode("utf-8"))
        
        # Format the response (you can adjust this based on what data you want to return)
        if 'organic' in data:
            results = []
            for item in data['organic'][:2]: 
                # Get first 3 results
                site_content = scrape_website(item.get('link', ''))
                results.append(f"Title: {item.get('title', '')}\nLink: {item.get('link', '')}\nContent: {site_content}\n")
            return "\n".join(results)
        return "No results found"
        
    except Exception as e:
        return f"Error performing search: {str(e)}"

def scrape_website(url: str) -> str:
    """
    Scrape text content from a website using BeautifulSoup

    Args:
        url: The URL of the website to scrape

    Returns:
        str: The cleaned text content from the website
    """
    try:
        import requests
        
        
        # Send GET request to the URL
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()  # Raise an exception for bad status codes
        
        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
            
        # Get text content
        text = soup.get_text()
        
        # Clean the text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text
        
    except Exception as e:
        return f"Error scraping website: {str(e)}"

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

def divide_two_numbers(a: float, b: float) -> float:
    """
    Divide first number by second number

    Args:
        a: The dividend (number to be divided)
        b: The divisor (number to divide by)

    Returns:
        float: The result of the division
    """
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a/b

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
                tools=[add_two_numbers, divide_two_numbers, search_serper]
            )
            
            # Process the tool calls
            available_functions = {
                'add_two_numbers': add_two_numbers,
                'divide_two_numbers': divide_two_numbers,
                'search_serper': search_serper
            }
            
            result = None
            if hasattr(response.message, 'tool_calls') and response.message.tool_calls:
                for tool in response.message.tool_calls:
                    function_to_call = available_functions.get(tool.function.name)
                    if function_to_call:
                        result = function_to_call(**tool.function.arguments)
            
            # Display the result
            if result is not None:
                summary = ollama.generate(
                    model='llama3.1:8b',
                    prompt=f"""
<SEARCH_RESULTS>
{result} 
</SEARCH_RESULTS>
from the above results create a summary of the information and answer the user's question.
use 10 bullets to summarize the information.
""",
                    stream=False
                )
                
                # answer = f"The sum is {result}"
                print(f"Assistant: {summary.response}")
            else:
                error_msg = "I couldn't calculate that. Please try asking me to add two numbers."
                print(f"Assistant: {error_msg}")
                
        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            print(f"Assistant: {error_msg}")

if __name__ == "__main__":
    main()