# OneTokenPy

I wanted an extremely simple, easy to use and yet powerful way to work with any LLM from Python. This is why I built OneTokenPy. It leverages Cosette from AnswerDotAI under the hood.

**In short:** OneTokenPy aims to be as simple as possible, but never limiting.

## ✨ Quick Demo

## 📦 Installation

```bash
pip install onetokenpy
```

**Optional (but strongly recommended if you use PDFs, PPTX, images, CSV):**

```bash
pip install requests pymupdf python-pptx pandas pillow-heif
```

## 🚀 Quick Start

The simplest way to use OneTokenPy is to just call the `ask` function.
`ask()` will will return a string. `ask()` will call the OpenRouter API with the default model using your `OPENROUTER_API_KEY` environment variable. If you don't have one, you can set it temporarily with `os.environ["OPENROUTER_API_KEY"] = "your_key_here"`. You can get a key from [openrouter.ai/settings/keys](https://openrouter.ai/settings/keys). To set it permanently, you can add the following to your `.bashrc` or `.zshrc` file:

> **Note:** You can also set the `OPENROUTER_API_KEY` environment variable permanently in your `.bashrc` or `.zshrc` file by simply adding the following line:`export OPENROUTER_API_KEY="your_key_here"`

Even without an api key, you can still use OneTokenPy. It will just use the default free model using a public api key managed by OneToken.AI. This is model can be slow and not so powerful. It is best to set your own api key.
```python
from onetokenpy import ask
ask("What is the capital of France?")

#> The capital of France is Paris.
```
### Using a different model

You can specify a different model by passing the `model` argument to `ask()`.


```python
ask(
    "How do I set the OPENROUTER_API_KEY environment variable permanently in linux/macos?",
    api_key="<PUT YOUR OWN API KEY HERE>"
)
```

Reminder: I you have not set the `OPENROUTER_API_KEY` environment variable, you can still use OneTokenPy. But you will have to pass the `api_key` argument to `ask()`.

```python
ask(
    "How do I set the OPENROUTER_API_KEY environment variable permanently in linux/macos?",
    model="anthropic/claude-3.7-sonnet",
    api_key="<PUT YOUR OWN API KEY HERE>"
)
```

---
## 📂 Context is Everything: Supercharge LLMs With External Data Called Attachments

**Give the model access to files, images, URLs, repositories and more!**

### Example: Passing a file or image to the model without using the `Attachment` object

We will explore all the ways to work with the `Attachment` object but first let's see how to do it without `Attachment` objects. Then we will see how `Attachment` objects facilitate the process of providing context to the model.

Let's we have the file `ot_example_file.txt` in the current directory.

> to create the file, you can use the following python code:
> ```python
> with open("ot_example_file.txt", "w") as f:
>     f.write("""
> This is an example file.
> It contains some example text.
> It is an example of a file that can be passed to a model.
> """)
> ```

Now we can pass the file to the model by passing the file path to the `ask` function.
```python
from onetokenpy import ask

ask(
    f"""What is in this file?
    {open("ot_example_file.txt").read()}
    """,
)
# → "This is an example file. It contains some example text. It is an example of a file that can be passed to a model."
```

Using the string interpolation is the lowest level way to pass a file to the model. This works for anything that can be converted to a string. For images we have to encode them to base64 and provide the 'next to' the prompt string. Like that:


```python
from onetokenpy import ask, Attachments
import base64

with open("cat_photo.jpg", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

ask(
    ["What is in this image?", encoded_string]
)
# → "This is a cat."
```

This is fine for simple cases and where you do not need a lot of prepprocessing, but things get quickly out of hand. This is why we have the `Attachments` object. For instance, attachments is great to ensure that the image is not too large for the model, it is also great even the context that you want to pass as a string requires lots of preprocessing, a prime example of that is if you want to pass the content of a PDF, word document, a website, a pptx, etc. All those must be converted into strings and images for the model to digest. Attachments handles all this for you with default handlers. And you can also add your own handlers.

### Example: Passing a file or image to the model using the `Attachments` object

Let's redo the previous example using the `Attachments` object.

```python
from onetokenpy import ask, Attachments

ask(
    "What is in this file?",
    attach=Attachments("ot_example_file.txt")
)
# → "This is an example file. It contains some example text. It is an example of a file that can be passed to a model."

ask(
    "What is in this image?",
    attach=Attachments("cat_photo.jpg")
)
# → "This is a cat."
```

### Example: Passing a website to the model

Here is an example of how to pass a website to the model. Notice that nothing is different, Attachments has handling rules and that detect from the string what to do with it (it is regex based). It this case, it is a URL and it will be fetched and the content will be passed to the model.

```python
ask(
    "Summarize this article:",
    attach=Attachments("https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)")
)
# → "This is a transformer."
```

### Multi-document context

You can easily pass more than one file to the model.

```python
ask(
    "What are the highlights and key numbers from all these files?",
    attach=Attachments("meeting_notes.md", "q2_report.pdf", "customers.csv")
)
```

### Supported context types out-of-the-box:

- Image files (`.jpg`, `.jpeg`, `.png`, `.webp`, `.heic`)
- Text files (`.txt`, `.md`)
- Word documents (`.docx`, with `python-docx`)
- Excel files (`.xlsx`, with `pandas`)
- PDFs (`.pdf`, with `pymupdf`)
- Presentations (`.pptx`, with `python-pptx`)
- CSVs (`.csv`, with `pandas`)
- URLs (auto-detects and fetches page/HTML, with `requests`)
- Repositories (`.git`, with `git`)
- Folders (`./folder`, with `os`)

Images, and text files will be provided as is completely unprocessed. If they are very large and may affect the performance of the model, Attachments will warn you and suggest optimizations.

For word, pdfs, presentations, and websites, a mix of text and images will be provided to the model.

You can always inspect the context that will be passed to the model by calling the `show` method on the `Attachments` object.

Excel files will be converted to CSVs and then to text. if there is more then 500 rows and 10 columns, a data summary will be provided instead.

For repositories or folders, a folder tree will be provided and optionally the content of the files will be provided. For this see: `Attachments.include_content`.

**Inspect your context:**

```python
attachments = Attachments("notes.txt", "plot.png")
attachments.show()
```

See the section about [defining your own handlers](#-extending-context-plug-in-your-own-handlers) for more information on `Attachments`.

---

## ⚒️ Tools: Let the LLM Call Your Python Functions

Tools for llms is a feature that allows the llm to trigger python functions and digest the results before responding to you. The tools must be running python functions. For instance, you can use the `calculator` function below to add, subtract, multiply, or divide two numbers.

> **Note:** The tool must be a python function. It cannot be a lambda function. It must have a docstring and it must have type hints.

> **Warning:** The tool will run in eval in python this, in theory let's the llm call any imported python function. You can thus require a user confirmation to gate the tool calls. In practice, with the right docstring, which acts as a prompt, the probability of the llm not using the calculator tools as a calculator is very low.

```python
def scientific_calculator(formula: str) -> float:
    """Evaluates a scientific formula.
    This will run in eval in python, ONLY USE THIS FOR SCIENTIFIC CALCULATIONS.
    You cannot import anything. 
    We have imported math, numpy, scipy and random for you.
    """
    import math
    import numpy as np
    import scipy
    import random

    print(f"Calculating: {formula}")
    # #user confirmation interactive input:
    # user_confirmation = input("Do you want to run this calculation? (y/n)")
    # if user_confirmation == "y":
    #     return eval(formula)
    # else:
    #     return "User cancelled the calculation."
    return eval(formula)

ask("What is so very cool calculation you can do with the scientific calculator? Describe what you did.", tools=[scientific_calculator])

# → Calculating: (1 + math.sqrt(5)) / 2
# → "What I calculated was the golden ratio using the formula (1 + √5)/2, which equals approximately 1.618033988749895. \n\nThis number is fascinating because:\n1. It appears in the growth patterns of plants, spiral arrangements in shells, and proportions considered aesthetically pleasing in art\n2. It's the limit of the ratio of consecutive Fibonacci numbers\n3. A rectangle with sides in this proportion (1:1.618...) is considered visually harmonious\n\nThe golden ratio has been used in design for centuries and continues to fascinate mathematicians and artists alike due to its unique mathematical properties and its seemingly universal presence in aesthetically pleasing proportions."
```

You can give any tools you can think of to the model. As long as they are python functions that runs in your python environment.

### Example: Python interpreter
Let's define a python interpreter tool.

```python
def python_interpreter(code: str, _globals={'__builtins__': __builtins__}):
    """Executes code in a python interpreter.
    This interpreter stays turn on as long as you or the user are not stoping it.
    This will run in eval in python
    """
    import io, contextlib, traceback
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        try:
            exec(code, _globals)
        except Exception:
            traceback.print_exc()
    return out.getvalue()

ask("Can you list the files in the current directory?",
    tools=[python_interpreter])

# → 'Here are the files and directories in your current directory:\n- .vscode (directory)\n- utils_llm.py\n- preproce\n- tst.txt\n- easyllm_gemini.py\n- utils_llm_before_newcontext.py\n- pyproject.toml\n- invoices (directory)\n- __pycache__ (directory)'
```

### Example: Memory
Let's do one last powerful example. Here we will create a 'memory' tool.

```python
import sqlite3

# Keep or initialize a persistent connection
_conn = sqlite3.connect('memory.db')
_conn.execute('CREATE TABLE IF NOT EXISTS memories (id INTEGER PRIMARY KEY AUTOINCREMENT, memory TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)')

def memory_tool(sql: str) -> str:
    """
    Execute a SQL query against a persistent SQLite 'memories' table.

    The tool manages a table named 'memories' within the 'memory.db' SQLite database.
    The table has the following schema (always present):

        memories (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            memory    TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )

    Usage:
      - Accepts a single SQL statement as a string argument.
      - Allowed operations: INSERT, SELECT, UPDATE, DELETE targeting the 'memories' table.
      - Schema modifications (e.g., DROP, ALTER), PRAGMA statements, multiple statements,
        and queries outside of 'memories' are not allowed.

    Args:
        sql (str): The SQL statement to execute (must target the 'memories' table).

            Examples:
                # Insert a new memory
                "INSERT INTO memories (memory) VALUES ('Meeting with Alice at 3pm')"

                # Retrieve memories containing the word 'Alice'
                "SELECT id, memory, timestamp FROM memories WHERE memory LIKE '%Alice%'"

                # Update a memory by id
                "UPDATE memories SET memory = 'Met Alice for coffee at 3pm' WHERE id = 1"

                # Delete a memory by id
                "DELETE FROM memories WHERE id = 1"

                # Get the 5 most recent memories
                "SELECT * FROM memories ORDER BY timestamp DESC LIMIT 5"

    Returns:
        str:
            - For SELECT queries: string representation of a list of rows (e.g., "[ (1, 'text', 'timestamp'), ... ]"),
              or 'No results found.' if no records match.
            - For INSERT, UPDATE, DELETE: 'Query executed successfully.'
            - On error (e.g., invalid query), returns "Error: <description>".

    Notes:
      - SQL statement must be a complete, valid command for the 'memories' table.
      - Output is always a string (not JSON).
      - Queries are synchronous and may block for long executions.
      - This tool is for demonstration purposes. Invalid or unsafe queries will return a descriptive error message.
    """
    try:
        cursor = _conn.execute(sql)
        if sql.strip().lower().startswith('select'):
            result = cursor.fetchall()
            return str(result) if result else 'No results found.'
        _conn.commit()
        return 'Query executed successfully.'
    except Exception as e:
        return f'Error: {e}'
```

If we simply call the memory tool, it will return the string 'No results found.'.

```python
memory_tool("SELECT * FROM memories")
# → 'No results found.'
```

We can new use to tool to tell the llms about our name.

```python
ask("Can you remember that my name is Maxime?", tools=[memory_tool])
# → "I've stored the information that your name is Maxime. This will help me remember your name in our future conversations."

ask("What is my name?", tools=[memory_tool])
# → 'Based on the stored memory, your name is Maxime.'
```

Notice the very big docstring? This is important as it tell the llm what to do with the tool.

## Composing tools and context

You can compose tools and context together.

```python
from onetokenpy import ask, Attachments

ask(
    "Please save the content of this file to my memory.",
    attach=Attachments("ot_example_file.txt"),
    tools=[memory_tool]
)
```

---
## 🗣️ Conversations: Stateful and Intuitive

`ask()`, `Attachments()` and `tools` are very powerful. But sometimes you may want to have a multi turn conversation with the llm. For this, you can use a `Chat` object and to create one you can use the `new_chat()` function.

`Chat` object are callable and will return a llm response not the Chat object itself. Let's see how to use it.

```python
from onetokenpy import new_chat
my_first_chat = new_chat()
my_first_chat("Hello, my name is Max!")
# → Hello Max! How can I assist you today?
my_first_chat("What did I just say?")
# → Your name is Max, as you mentioned in your introduction.
```

Notice that we do not need the memory tool here. The chat object remembers the context but another chat object will not.

```python
my_second_chat = new_chat()
my_second_chat("What is my name?")
# → 'I don't know your name.'
```

Chats also have a history.

```python
my_first_chat.h
# → [{'role': 'user', 'content': 'Hello, my name is Max!'},
# →  ChatCompletionMessage(content='Hello Max! How can I assist you today?', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None, reasoning=None),
# →  {'role': 'user', 'content': 'What is my name?'},
# →  ChatCompletionMessage(content='Your name is Max, as you mentioned in your introduction.', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None, reasoning=None)]
```

You can easily manipulate the history.

```python
my_first_chat = new_chat()
my_first_chat("Hello, my name is Max!")
# → Hi Max! It's nice to meet you. How can I help you today?

my_first_chat("What is my name?")
# → Your name is Max, as you mentioned in your introduction.

my_first_chat.h[-1] = {"role": "assistant", "content": "You stink!!"}
my_first_chat("What did you say! :O")
# → I apologize for that inappropriate response. That was completely unacceptable and doesn't reflect how I should communicate. Your name is Max, as you introduced yourself earlier. I'm sorry for the disrespectful message. How can I assist you properly today?

my_first_chat.h[-3:]
# →  [{'role': 'assistant', 'content': 'You stink!!'}, {'role': 'user', 'content': 'What did you say! :O'}, ChatCompletionMessage(content="I apologize for that inappropriate response. That was completely unacceptable and doesn't reflect how I should communicate.\n\nYour name is Max, as you introduced yourself earlier. I'm sorry for the disrespectful message. How can I assist you properly today?", refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None, reasoning=None)]
```

### Chat Attachments and tools  

Just like with `ask()`, you can pass attachments and tools to the chat.

```python
from onetokenpy import new_chat, Attachments

conversation = new_chat(
    attach=Attachments("ot_example_file.txt"),
    tools=[scientific_calculator])
conversation("What is the square root of 5? and what is in the file?")
# → The square root of 5 is approximately 2.23606797749979.
# → The file contains the text "This is an example file. It contains some example text. It is an example of a file that can be passed to a model."
```

## 🏗️ Compose, Edit, and Hack Your Context

If you save the attachments object, you can edit it.

```python
attachments = Attachments("ot_example_file.txt")
attachments.remove("ot_example_file.txt")
attachments.add("more_data.csv")
```

---


## 🏷️ Extending Context: Plug in Your Own Handlers

Support any exotic data:

```python
from onetokenpy import ask, Attachments
import yaml

@Attachments.register_handler(".yaml")
def yaml_handler(path):
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return {
        "type": "text",
        "content": f"YAML keys: {', '.join(data.keys())}",
        "identifier": str(path),
    }

ask("What keys are defined here?", attach=Attachments("config.yaml"))
```

---

## 🔍 Model Selection: Pick the Best LLM for Each Task

`ask()` and `new_chat()` will use the default model. But you can specify a different model. For this you have to provide the correct openrouter model id. This can be cumbersome to remember. This is why we have the `llm_picker()` function. `llm_picker()` will return a list of model ids that match the prompt based on the openrouter model list.

```python
from onetokenpy import llm_picker

models = llm_picker("I want the latest model from google")
print(models)
# → ['google/gemini-2.5-pro-preview']

ask("Who trained you and what is your knowledge cutoff date?", model=models[0])
# → 'I am a large language model, trained by Google. My knowledge cutoff is **June 2024**. Therefore, I cannot provide you with any information about events or developments that have occurred since that time.'
```

---

## 🏁 What's Next?

1. Fire up a `new_chat()` and converse.
2. Give it something non-trivial (like an image, PDF, or a URL!).
3. Register a homemade context handler for your favorite data type.
4. Build a workflow with custom Python tools.
5. Try `llm_picker()` and experiment with diverse LLMs.


Happy LLMing! 🚀  

## 📝 TO DO

- [ ] Add Azure and Bedrock providers directly for enterprise users.

