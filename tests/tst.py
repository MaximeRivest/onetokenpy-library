#%%
import onetokenpy as ot
#%%
ot.ask("What is the capital of France?")

#%%
ot.ask("What is the capital of France?", cli=ot.anthropic_client(), model="claude-3-5-haiku-20241022")


#%%
ot.core.DEFAULT_CLIENT = ot.anthropic_client()
ot.core.DEFAULT_MODEL = "claude-3-5-haiku-20241022"
#%%

ot.ask("What is the capital of France?")

#%%
import cosette
import openai

ot.core.DEFAULT_MODEL = "deepseek-r1:1.5b"
ot.core.DEFAULT_CLIENT = cosette.Client(
        model=ot.core.DEFAULT_MODEL,
        cli=openai.OpenAI(
            base_url = 'http://localhost:11434/v1',
            api_key='ollama', # required, but unused
        )
    )



ot.ask("What is the capital of France?")


#%%
import onetokenpy as ot
import math
import numpy as np
import scipy
import random

def scientific_calculator(formula: str) -> float:
    """Evaluates a scientific formula.
    This will run in eval in python, ONLY USE THIS FOR SCIENTIFIC CALCULATIONS.
    You cannot import anything. 
    We have imported math, numpy, scipy and random for you.
    ```python
    import math
    import numpy as np
    import scipy
    import random
    ```
    If error try again! Remeber to use python syntax coherent with the above imports.

    example:
    ```python
    math.sqrt(2)
    np.sin(np.pi/2)
    random.randint(1, 100)
    ```
    """
    

    print(f"Calculating: {formula}")

    try:
        return eval(formula)
    except Exception as e:
        return f"Error: {e}"

ot.ask("What is so very cool calculation you can do with the scientific calculator? Do it! Then describe what you did.", tools=[scientific_calculator])
# %%
import onetokenpy as ot
ctx = ot.Attachments("/home/maxime/Projects/onetokenpy-library/onetokenpy/helpers.py")
ot.ask("Did I provide information about a website?",
         attach=ctx)
# %%md
# test the chat
import onetokenpy as ot
chat = ot.new_chat()
chat("What is the capital of France?")
# %%
chat1 = ot.new_chat()
chat1("my name is maxime")
#%%
chat1("what is my name?")
# %%
chat1.model = "anthropic/claude-3.5-haiku"
# %%
chat1("who trained you and what is your knowledge cutoof?")
# %%
c = chat1._cli
#
chat1.__dict__
# %%
# list all possible methods
dir(c)
# %%
# list all possible attributes
c.__dict__
# %%
c.c.__dict__
# %%
from onetokenpy import new_chat, Attachments
chat = new_chat(attach=Attachments("https://answerdotai.github.io/cosette/llms.txt"))
chat("What is Cosette?")
# %%
chat.h
# %%
from onetokenpy import new_chat, Attachments
attach = Attachments("https://answerdotai.github.io/cosette/llms.txt")
attach.show()
# %%
for i in attach.to_model_format("anthropic/claude-3.5-haiku"):
    print(i)
# %%

import onetokenpy as ot
ot.ask("What is in this image?",
       attach=ot.Attachments("/home/maxime/Pictures/Screenshot from 2020-07-04 00-42-04.png"))
# %%
import onetokenpy as ot
import claudette
import anthropic


c = claudette.Client(
        model="claude-3-7-sonnet-latest", # This sets cosette.Client.model
        cli=anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY")
        )
    )
c("hello")

c
# %%
