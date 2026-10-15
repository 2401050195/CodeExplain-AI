
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import requests

app = FastAPI(title="CodeExplain AI")

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma3:1b"


class CodeRequest(BaseModel):
    code: str


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport"
              content="width=device-width, initial-scale=1.0">
        <title>CodeExplain AI</title>

        <style>
            body {
                font-family: Arial, sans-serif;
                background: #f3f6fc;
                margin: 0;
                padding: 30px 16px;
                color: #1f2937;
            }

            .container {
                max-width: 850px;
                margin: auto;
                background: white;
                padding: 28px;
                border-radius: 16px;
                box-shadow: 0 5px 24px #00000012;
            }

            h1 { color: #3157c8; }

            textarea {
                width: 100%;
                min-height: 190px;
                box-sizing: border-box;
                padding: 14px;
                border: 1px solid #cbd5e1;
                border-radius: 8px;
                font-family: Consolas, monospace;
                font-size: 15px;
                resize: vertical;
            }

            button {
                margin-top: 14px;
                padding: 12px 20px;
                background: #3157c8;
                color: white;
                border: none;
                border-radius: 8px;
                cursor: pointer;
                font-size: 16px;
            }

            button:disabled {
                opacity: 0.6;
                cursor: wait;
            }

            #result {
                white-space: pre-wrap;
                overflow-wrap: anywhere;
                line-height: 1.7;
                background: #f8fafc;
                padding: 16px;
                border-radius: 8px;
                min-height: 60px;
            }

            .muted { color: #64748b; }
        </style>
    </head>

    <body>
        <div class="container">
            <h1>CodeExplain AI</h1>
            <p class="muted">
                Understand Python code with help from local AI.
            </p>

            <h3>Paste your Python code</h3>

            <textarea id="code" placeholder="Example:
for i in range(1, 6):
    print(i)"></textarea>

            <br>
            <button id="explainBtn" onclick="explainCode()">
                Explain My Code
            </button>

            <h3>AI Explanation</h3>
            <div id="result" aria-live="polite">
                Your explanation will appear here.
            </div>
        </div>

        <script>
            async function explainCode() {
                const code = document.getElementById("code").value;
                const result = document.getElementById("result");
                const button = document.getElementById("explainBtn");

                if (!code.trim()) {
                    result.textContent = "Please enter some Python code.";
                    return;
                }

                button.disabled = true;
                result.textContent = "Analyzing your code...";

                try {
                    const response = await fetch("/explain", {
                        method: "POST",
                        headers: {
                            "Content-Type": "application/json"
                        },
                        body: JSON.stringify({code: code})
                    });

                    const data = await response.json();

                    if (!response.ok) {
                        throw new Error(data.detail || "Request failed");
                    }

                    result.textContent = data.explanation;
                } catch (error) {
                    result.textContent = "Error: " + error.message;
                } finally {
                    button.disabled = false;
                }
            }
        </script>
    </body>
    </html>
    """


@app.post("/explain")
def explain_code(request: CodeRequest):
    code = request.code.strip()

    if not code:
        raise HTTPException(
            status_code=400,
            detail="Please enter some Python code."
        )

    if len(code) > 8000:
        raise HTTPException(
            status_code=400,
            detail="Please keep your code under 8000 characters."
        )

    prompt = f"""
You are CodeExplain AI, a patient Python tutor for first-year
B.Tech students.

Explain the submitted Python code in simple English.

Use these sections:
1. Purpose: What does the program do?
2. Line-by-line explanation: Explain each important line.
3. Execution flow: Explain the order in which it runs.
4. Output: Show the expected output, if it can be determined.
5. Key concepts: Explain the Python concepts used.

If the code contains an error, explain the likely issue and
suggest a correction.

Do not execute the submitted code. Analyze it as text.
If the output depends on unknown input or external conditions,
explain that clearly.

Python code:
<user_code>
{code}
</user_code>
"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False
            },
            timeout=180
        )

        response.raise_for_status()
        result = response.json()
        explanation = result.get("response", "").strip()

        if not explanation:
            raise HTTPException(
                status_code=502,
                detail="The AI returned an empty explanation."
            )

        return {"explanation": explanation}

    except requests.exceptions.Timeout:
        raise HTTPException(
            status_code=504,
            detail="The AI took too long. Please try again."
        )

    except requests.exceptions.RequestException:
        raise HTTPException(
            status_code=503,
            detail="Cannot connect to Ollama. Check whether it is running."
        )
