# %%md
# # How to create a model merging chat?
# 

# %%md
# We will first define the list of models we want to use and merge.

# %%md
# now we will define one chat object, we will then clone it for each model and call it.
# Then we will take the last element of .h in the history list and prodive them all with the common history to a
# ask() function.

# %%
import onetokenpy as ot
from IPython.display import Markdown
from copy import deepcopy
import threading

def prompt_and_merge_chat(prompt, chat, list_of_models):
    responses = [None] * len(list_of_models)  # To store responses in order
    threads = []

    # Helper function to call a model and store its response
    def call_model_and_store_response(model_name, index, base_chat, current_prompt):
        tmp_chat_branch = deepcopy(base_chat)
        tmp_chat_branch.model = model_name
        try:
            res = ot.contents(tmp_chat_branch(current_prompt))
            responses[index] = res
        except Exception as e:
            # Store the exception or a placeholder if a call fails
            responses[index] = f"Error with model {model_name}: {e}"


    for i, model in enumerate(list_of_models):
        thread = threading.Thread(target=call_model_and_store_response, args=(model, i, chat, prompt))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()  # Wait for all threads to complete
    
    merged_response = chat(
        f"""You have been provided with a set of responses from various models to the latest user query: "{prompt}"
Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.
Responses from models: {responses}

Do not refer to the models in your response, just use then for crafting the best answer to the user query.""")
    chat.h[-2] = {"role":"user", "content":prompt}
    chat.h[-1] = {"role":"assistant", "content":ot.contents(merged_response)}
    return Markdown(chat.h[-1]["content"])

# # %%
# list_of_models = [
#     "anthropic/claude-3.5-haiku",
#     "openai/gpt-4.1",
#     "x-ai/grok-3-mini-beta",
#     "google/gemini-2.5-flash-preview"
# ]

# chat = ot.new_chat(model="x-ai/grok-3-mini-beta")
# prompt_and_merge_chat("Tell me a joke", chat, list_of_models)


# # %%
# chat.h

# # %%
# from onetokenpy import new_chat, Attachments
# list_of_models = [
#     "anthropic/claude-3.5-haiku",
#     "openai/gpt-4.1",
#     "x-ai/grok-3-mini-beta",
#     "google/gemini-2.5-flash-preview"
# ]
# chat = new_chat(attach=Attachments("https://answerdotai.github.io/cosette/llms.txt"))
# prompt_and_merge_chat("What is cosette?", chat, list_of_models)

# # %%
# prompt_and_merge_chat("Interesting! Can cosette refer to anything else?",
#                        chat, list_of_models)



