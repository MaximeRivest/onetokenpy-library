# %%md
# # Recipe Helper: From Ingredients to Delicious Meals
#
# OneTokenPy can power helpful AI assistants. This example demonstrates how to create a Recipe Helper
# that suggests meals based on your ingredients, whether you provide them as text or as a photo.
# We'll use a `Chat` object to allow for follow-up questions and refinements.
#
# This showcases a practical application combining conversational AI with the ability
# to process different types of input.

# %%
from onetokenpy import new_chat, Attachments
import claudette
from IPython.display import Markdown
import os # For managing example files

# Initialize a chat session for our recipe helper
# You can specify a model, or use the default. Let's pick a fast one for the example.
recipe_chat = new_chat(model="google/gemini-2.5-flash-preview") # Or try "anthropic/claude-3.5-haiku"

# %%md
# ## Scenario 1: Providing ingredients as a text list

# %%
ingredients_text = "chicken breast, broccoli, carrots, soy sauce, ginger, garlic, sesame oil"
prompt1 = f"I have these ingredients: {ingredients_text}. Can you suggest a healthy and simple stir-fry recipe with step-by-step instructions and estimated prep/cook times?"

recipe_from_text = recipe_chat(prompt1)
recipe_from_text
# Expected output might be a stir-fry recipe.

# %%md
# ## Scenario 2: Providing ingredients as a photo
# For this example, we'll simulate having an image file.
# In a real application, 'my_pantry_items.jpg' would be an actual image of your ingredients.

# %%
image_filename = "my_pantry_items.jpg"
image_description_for_dummy_file = "Placeholder content for an image of, e.g., bell peppers, onion, eggs, and cheese."

try:
    # Create a dummy file that simulates an image file for the example
    with open(image_filename, "w") as f:
        f.write(image_description_for_dummy_file)
    print(f"Created dummy image file: {image_filename} (describing: '{image_description_for_dummy_file}')")

    prompt2 = "I took a picture of some items I have (see attached). Based on what you can identify, what breakfast dish could I make? Please list the ingredients you see and provide a simple recipe."

    # The Chat object can take attachments directly with the prompt
    recipe_from_image = recipe_chat(
        prompt2,
        attach=Attachments(image_filename)
    )
    print("\nAI's recipe suggestion (from image):")
    display(Markdown(recipe_from_image))
    # Expected output might be an omelette recipe if the placeholder description is good.

finally:
    # Clean up the dummy image file
    if os.path.exists(image_filename):
        os.remove(image_filename)
        print(f"\nCleaned up dummy image file: {image_filename}")

# %%md
# ## Scenario 3: Follow-up questions
# The chat remembers the previous interactions.
# Let's ask a follow-up related to the last recipe generated (from the image).

# %%
follow_up_prompt = "That breakfast recipe sounds good! Can you suggest a variation that includes spinach, if I have some?"
refined_recipe = recipe_chat(follow_up_prompt)
print("AI's refined recipe suggestion:")
display(Markdown(refined_recipe))
# Expected output might be the previous recipe adjusted to include spinach.

# %%md
# You can always access the conversation history to see how the interaction unfolded:

# %%
print("\n--- Conversation History Snippet ---")
for i, message in enumerate(recipe_chat.h):
    content_preview = str(message.get('content', 'N/A'))
    # For display purposes, we might want to shorten very long content strings
    if isinstance(content_preview, str) and len(content_preview) > 150:
        content_preview = content_preview[:150] + "..."
    print(f"Message {i+1} ({message['role']}): {content_preview}")

# %%md
# This example shows how to use `new_chat` for a multi-turn conversation,
# pass textual information directly, and use `Attachments` for image-based input.
# The follow-up question demonstrates the conversational memory of the `Chat` object.