# Hackathon 2025 Template

## Business Problem

## Submission Guidelines:
1. Do NOT upload your API key to your Github Repo
2. Do NOT upload any data to the Github Repo
3. You will not be able to push any last updates past the submission deadline. Make sure you plan accordingly!

### Grounds for Disqualification
1. If any API keys are found in your final submitted Github Repo, you will be disqualified.
2. If you upload the entirety of the data given to the final submitted Github Repo, you will be disqualified.

We understand mistakes happen. Thus, if any of the above happens, that's okay, just make sure that you remove them from your repo before the deadline.


## Communications
Communication will primarily happen through Discord: https://discord.gg/9U29z5Qh

## Data
Link to the data is here as well as on Discord: 

## How to Use an LLM API key.
**IMPORTANT**: You should NEVER hardcode your API key into your file.
1. You should keep your API key in a `.env` file
2. You should set your `.gitignore` to contain any files you don't want uploaded to github (`.env`, `data/`, `output/`, etc.)

`.env` should contain:

```
OPENAI_API_KEY=your-api-key-here
GEMINI_API_KEY=your-api-key-here
```

`.gitignore` should contain:

```
.env
data/
output/
*.ipynb
```

### Chatgpt
```python
from openai import OpenAI
import os

# Initialize the client with your API key
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")  # or set directly: api_key="your-api-key-here"
)

# Make a query
response = client.chat.completions.create(
    model="gpt-5-mini",  # or "gpt-5" for the full model
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is machine learning?"}
    ]
)

# Print the response
print(response.choices[0].message.content)

```

### Gemini
```python
from google import genai
from google.genai import types
import os

# Initialize the client with your API key
client = genai.Client(
    api_key=os.environ.get("GEMINI_API_KEY")  # or set directly: api_key="your-api-key-here"
)

# Make a query using Gemini 2.5 Flash
response = client.models.generate_content(
    model="gemini-2.5-flash",  # or "gemini-2.5-pro" for better reasoning
    contents=[
        {
            "role": "user",
            "parts": [{"text": "What is machine learning?"}]
        }
    ],
    config=types.GenerateContentConfig(
        system_instruction="You are a helpful assistant."
    )
)

# Print the response
print(response.text)
```

Good luck and have fun!
