
## OneTokenPy – Minimal Starter Cheat-Sheet for Python AI Experts

### 1. Installation

```bash
pip install onetokenpy
# For attachments: (PDF, PPTX, images, CSV, etc.)
pip install requests pymupdf python-pptx pandas pillow-heif
```

---

### 2. Authentication

Set your OpenRouter API key (recommended):
```bash
export OPENROUTER_API_KEY=YOUR_KEY   # Or set via os.environ or pass as api_key param
```

---

### 3. Basic LLM Prompt

```python
from onetokenpy import ask

# Basic call (uses default/fallback model if api_key/env not set)
reply = ask("What is the capital of France?")

# Specify model and/or API key if desired
reply = ask("Prompt", model="anthropic/claude-3.7-sonnet", api_key="YOUR_KEY")
```

---

### 4. Attachments (Bring in Files/Images/URLs/Repos/Folders)

```python
from onetokenpy import ask, Attachments

# Single file
reply = ask("What's in this file?", attach=Attachments("file.txt"))
# Image, PDF, website, etc.:
reply = ask("Describe this image", attach=Attachments("cat.jpg"))
reply = ask("Summarize article", attach=Attachments("https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)"))
# Multiple/context-rich:
reply = ask("Key takeaways?", attach=Attachments("notes.md", "report.pdf", "site.com"))
```

*Inspect context pre-processing:*
```python
Attachments("file.txt", "image.png").show()
```

*Extend with custom handler:*
```python
from onetokenpy import Attachments
import yaml

@Attachments.register_handler(".yaml")
def _(path):
    with open(path) as f: data = yaml.safe_load(f)
    return {"type":"text","content":f"YAML keys: {', '.join(data.keys())}"}
```

---

### 5. Tools (LLM-Initiated Function Calls)

```python
def calculator(expr: str) -> float:
    """Evaluates Python math expressions."""
    import math
    return eval(expr)

reply = ask("What is the golden ratio?", tools=[calculator])
```

---

### 6. Stateful Multiturn Chats

```python
from onetokenpy import new_chat

chat = new_chat()  # Optionally pass: model, api_key, attach, tools
chat("My name is Max.")
chat("What did I just say?")
# chat.h → chat history
```

---

### 7. Model Selection

```python
from onetokenpy import llm_picker

models = llm_picker("latest, powerful google model")
reply = ask("Prompt!", model=models[0])
```

---

## Suggestions / Next Steps

- **Compose Attachments** for files, URLs, images for context-enriched prompts.
- **Create custom tools** (Python functions with docstrings & type hints).
- **Spin up `new_chat()`** for multi-turn, context-tracking conversations.
- **Explore & extend Attachments** with `.register_handler()` for custom data types.
- **Inspect and refine**: Use `Attachments.show()`, review chat history with `chat.h`.

---

**That’s it! This is all you need to get started with OneTokenPy’s power features—see docstrings and source for deep diving.**