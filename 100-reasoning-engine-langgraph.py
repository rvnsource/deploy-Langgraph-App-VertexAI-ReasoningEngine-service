# Set the environment variable to point to your service account key
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/ravi/.config/gcloud/genai-443318-637e44e3cf32.json"


PROJECT_ID = "genai-443318"
LOCATION = "us-central1"
STAGING_BUCKET = "gs://reasoning-bucket-2"

import vertexai
vertexai.init(project=PROJECT_ID, location=LOCATION, staging_bucket=STAGING_BUCKET)

########
# Tools
########
from langchain_core.tools import tool
@tool
def get_product_details(product_name: str):
    """Gathers basic details about a product"""
    details = {
        "smartphone": "A smartphone is a mobile phone with advanced computing capabilities.",
        "speaker": "A speaker is an electroacoustic transducer that converts electrical audio signals into sound waves.",
        "headphones": "Headphones are audio devices worn over or around the ears.",
        "shoes": "Shoes are footwear designed to protect and comfort the human foot."
    }
    return details.get(product_name, "Product details not found")


import requests
@tool
def get_weather(city: str):
    """Get weather details for the given city"""

    latitude, longitude = get_lat_long(city)
    api_url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,relativehumidity_2m,weathercode&hourly=temperature_2m,relativehumidity_2m,weathercode&daily=weathercode&temperature_2m_max,temperature_2m_min&windspeed_10m_max&precipitation_sum&rain_sum&showers_sum&snowfall_sum"

    try:
        response = requests.get(api_url)
        response.raise_for_status()  # Raise an exception for bad status codes
        data = response.json()

        current_weather = data["current"]
        temperature = current_weather["temperature_2m"]
        humidity = current_weather["relativehumidity_2m"]
        weather_code = current_weather["weathercode"]

        # Map weather code to a descriptive string (optional)
        weather_descriptions = {
            0: "Clear sky",
            1: "Mainly clear",
            2: "Partly cloudy",
            3: "Overcast",
            45: "Fog",
            48: "Depositing rime fog",
            51: "Light drizzle",
            53: "Moderate drizzle",
            55: "Dense drizzle",
            56: "Light freezing drizzle",
            57: "Dense freezing drizzle",
            61: "Slight rain",
            63: "Moderate rain",
            65: "Heavy rain",
            66: "Light freezing rain",
            67: "Heavy freezing rain",
            71: "Slight snow fall",
            73: "Moderate snow fall",
            75: "Heavy snow fall",
            77: "Snow grains",
            80: "Slight rain showers",
            81: "Moderate rain showers",
            82: "Violent rain showers",
            83: "Slight snow showers",
            84: "Heavy snow showers",
            85: "Slight snow flurries",
            86: "Heavy snow flurries",
            95: "Slight or moderate thunderstorm",
            96: "Thunderstorm with slight hail",
            99: "Thunderstorm with heavy hail"
        }

        weather_description = weather_descriptions.get(weather_code, "Unknown")

        return f"temperature: {temperature} humidity: {humidity} description: {weather_description} in {city} - Latitude {latitude}, Longitude {longitude}"

    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return f"Error fetching weather data: {e}"


import requests
from geopy.geocoders import Nominatim

def get_lat_long(city):
    """
    Gets the latitude and longitude of a given city using the Nominatim geocoder.

    Args:
        city (str): The name of the city.

    Returns:
        tuple: A tuple containing the latitude and longitude, or None if the city is not found.
    """
    geolocator = Nominatim(user_agent="my_geocoder")
    location = geolocator.geocode(city)

    if location:
        return location.latitude, location.longitude
    else:
        return None


#########
# Router
#########
from typing import List, Literal
from langchain_core.messages import BaseMessage, HumanMessage
def router(state: List[BaseMessage]) -> Literal["get_product_details", "get_weather", "__end__"]:
    """Initiates product details retrieval if the user asks for a product"""
    # Get the tool_calls from the last message in the conversation history
    tool_calls = state[-1].tool_calls

    # If there are any tool_calls
    if len(tool_calls):
        # Return the name of the tool to be called
        tool_name = tool_calls[0]['name']
        return tool_name
        #return "get_product_details"
    else:
        return "__end__"

########################
# LangGraph Application
########################
from langchain_google_vertexai import ChatVertexAI
from langgraph.graph import END, MessageGraph
from langgraph.prebuilt import ToolNode

class SimpleLangGraphApp:
    def __init__(self, project: str, location: str) -> None:
        self.project_id = project
        self.location = location

    def set_up(self) -> None:
        model = ChatVertexAI(model="gemini-1.5-pro")
        model_with_tools = model.bind_tools([get_product_details, get_weather])

        builder = MessageGraph()
        builder.add_node("tools", model_with_tools)

        tool_node = ToolNode([get_product_details])
        builder.add_node("get_product_details", tool_node)
        builder.add_edge("get_product_details", END)


        tool_node2 = ToolNode([get_weather])
        builder.add_node("get_weather", tool_node2)
        builder.add_edge("get_weather", END)

        builder.set_entry_point("tools")
        builder.add_conditional_edges("tools", router)

        self.runnable = builder.compile()

    # The query method will be used to send inputs to the agent
    def query(self, message: str):
        """Query the application.

        Args:
            message (str): The user message.

        Returns:
            str: The LLM response.
        """
        chat_history = self.runnable.invoke(HumanMessage(message))

        return chat_history[-1].content


################
# Local Testing
################
agent = SimpleLangGraphApp(project=PROJECT_ID, location=LOCATION)
agent.set_up()

print(agent.query(message="Get product details for shoes."))
print(agent.query(message="Get product details for headphones."))
print(agent.query(message="Tell me about the weather in New Delhi."))
print(agent.query(message="Tell me about the weather in Coimbatore."))
print(agent.query(message="Tell me about the weather in New York."))
print(agent.query(message="Write python code to add two numbers"))


#####################################################
# Deploy the app into GCP Vertex AI Reasoning Engine
#####################################################
from vertexai.preview import reasoning_engines
remote_agent = reasoning_engines.ReasoningEngine.create(
    SimpleLangGraphApp(project=PROJECT_ID, location=LOCATION),
    requirements=[
        "google-cloud-aiplatform[langchain,reasoningengine]",
        "langgraph",
        "geopy"
    ],
    display_name="My AI agent for product and weather details",
    description="My AI agent for product and weather details",
    extra_packages=[]
)