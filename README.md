# Hackathon 2025 Template

## Submission Guidelines:
1. Do NOT upload your API key to your Github Repo. This is very bad practice. If any API keys are found after the deadline, you will be disqualified!
3. Do NOT upload all of the data to the Github Repo. It's bad practice.
4. You will not be able to push any last updates past the submission deadline. Make sure you plan accordingly!
5. Please make sure that your readme includes a brief description of what your code does as well as instructions on how to run your code.
6. Please do not set your repos to public until after the presentations!

## Communications
Communication will primarily happen through Discord: https://discord.gg/9U29z5Qh

## Data
Link to the data is here as well as on Discord: https://www.kaggle.com/datasets/davidcariboo/player-scores

Link to the data dictionary is here as well as on Discord: https://docs.google.com/document/d/1AvAvjpTnqF_sav1Zg5wq36RS-8uyjyCQRlzwSHX57Vs/edit?usp=drive_link

## Git cloning
To download this repository and push/pull from here, you will need `git` installed on your device ([https://git-scm.com/install/](https://github.com/git-guides/install-git)).

Git clone (first downloada of the Github repo)
```
git clone <your-repository-url-here>
```

Git Push (push updates to Github Repo
```
# 1. Stage all your changes
git add .

# 2. Commit your changes with a clear message
git commit -m "Added my new feature"

# 3. Push your commits to GitHub
git push
```

Git Pull (pull updates from Github Repo)
```
git pull
```

## LLMs
ChatGPT API keys will be provided (1 per team). Please contact **mateusz** in Discord to ask for the key.

### How to keep an LLM API key safe from bad actors.
**IMPORTANT**: You should NEVER hardcode your API key into your file. I have included an example `.env.example` and `.gitignore` to get you started.
1. You should rename `.env.example` to `.env` keep your API key in a `.env` file.
2. You should set your `.gitignore` to contain any files you don't want uploaded to github (`.env`, `data/`, `output/`, etc.).

`.env` example:

```
OPENAI_API_KEY=your-api-key-here
GEMINI_API_KEY=your-api-key-here
```

`.gitignore` example:

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
Notably: Gemini has a free tier for those with Google Accounts (https://ai.google.dev/gemini-api/docs/rate-limits)
- 50 Responses per day for Gemini-2.5-pro
- 250 Responses per day for Gemini-2.5-flash

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
