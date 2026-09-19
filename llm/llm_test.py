from google import genai

client = genai.Client()
# creates a client to communicate with google's gemini API.
# this client automatically takes in API credentials from environment. 
# Or can be passed manually as:
# client = genai.Client(api_key="API_KEY")

interaction = client.interactions.create(
    model="gemini-3.5-flash-lite",
    input="Explain me how AI works in detail."
)
print(interaction.output_text)