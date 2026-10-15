import requests

url = "http://localhost:11434/api/generate"

data = {
    "model": "gemma3:1b",
    "prompt": "Explain this code to a beginner: print(2 + 3)",
    "stream": False
}

response = requests.post(url, json=data, timeout=120)
response.raise_for_status()

result = response.json()
print(result["response"])