# import requests
from ollama import chat
from ollama import ChatResponse


# def get_weather_in_tel_aviv() -> dict:
#     """
#     Fetch the current weather in Tel Aviv using the Open-Meteo API.

#     Returns:
#         dict: A dictionary containing the weather data.
#     """
#     url = "https://api.open-meteo.com/v1/forecast"
#     params = {
#         "latitude": 32.0853,  # Latitude for Tel Aviv
#         "longitude": 34.7818,  # Longitude for Tel Aviv
#         "current_weather": True,
#     }
#     response = requests.get(url, params=params)
#     if response.status_code == 200:
#         return response.json().get("current_weather", {})
#     else:
#         return {"error": f"Failed to fetch weather data. Status code: {response.status_code}"}


def get_weather(city: str) -> dict:
    """
    Get weather information for specific cities

    Args:
        city (str): The city name (Tel Aviv or New York)

    Returns:
        dict: Weather information for the specified city
    """
    match city.lower():
        case "tel aviv":
            return {
                "temperature": 28,
                "condition": "sunny",
                "humidity": 65,
                "description": "Hot and sunny Mediterranean weather"
            }
        case "new york":
            return {
                "temperature": 20,
                "condition": "partly cloudy",
                "humidity": 55,
                "description": "Mild with some clouds"
            }
        case _:
            return {
                "error": f"Weather information not available for {city}. Only Tel Aviv and New York are supported."
            }


# Define the tools for fetching weather
get_weather_tool = {
    'type': 'function',
    'function': {
        'name': 'get_weather',
        'description': 'Get weather information for Tel Aviv or New York',
        'parameters': {
            'type': 'object',
            'required': ['city'],
            'properties': {
                'city': {
                    'type': 'string',
                    'description': 'The city name (Tel Aviv or New York)',
                    'enum': ['Tel Aviv', 'New York']
                }
            }
        }
    }
}

messages = [{'role': 'user', 'content': 'what is the main street in Tel Aviv?'}]
print('Prompt:', messages[0]['content'])

available_functions = {
    'get_weather': get_weather,
}

response: ChatResponse = chat(
    'qwen2.5-coder:7b',
    messages=messages,
    tools=[get_weather_tool],
)

if response.message.tool_calls:
    # There may be multiple tool calls in the response
    for tool in response.message.tool_calls:
        # Ensure the function is available, and then call it
        if function_to_call := available_functions.get(tool.function.name):
            print('Calling function:', tool.function.name)
            print('Arguments:', tool.function.arguments)
            output = function_to_call(**tool.function.arguments)
            print('Function output:', output)
        else:
            print('Function', tool.function.name, 'not found')

# Only needed to chat with the model using the tool call results
if response.message.tool_calls:
    # Add the function response to messages for the model to use
    messages.append({'role': 'tool', 'content': str(output), 'name': tool.function.name})

    # Get final response from model with function outputs
    final_response = chat('qwen2.5-coder:7b', messages=messages)
    print('Final response:', final_response.message.content)

else:
    print('No tool calls returned from model')