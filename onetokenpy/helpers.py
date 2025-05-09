"""
OneTokenPy.helpers - Utility helpers for model selection and compression.
"""

import requests
from .core import ask

def compress_openrouter_models(models_data):
    """
    Compresses the OpenRouter models JSON to reduce token consumption while preserving 
    decision-critical information for model selection.

    Args:
        models_data (dict): Raw JSON response from OpenRouter API

    Returns:
        dict: Compressed data with metadata about compression
    """
    if not models_data or 'data' not in models_data:
        return {'error': 'Invalid or empty model data', 'models': []}

    original_size = len(str(models_data))
    compressed_models = []

    for model in models_data.get('data', []):
        # Skip invalid entries
        if not isinstance(model, dict) or 'id' not in model:
            continue

        # Extract provider from ID for easier filtering
        provider = model['id'].split(
            '/')[0] if '/' in model['id'] else 'unknown'

        # Extract architecture info
        arch = model.get('architecture', {})
        modalities = arch.get('input_modalities', [])

        # Generate capability tags for efficient filtering
        tags = []
        if 'image' in modalities:
            tags.append("vision")
        if model.get('context_length', 0) >= 100000:
            tags.append("long-context")
        if 'tools' in model.get('supported_parameters', []):
            tags.append("tool-use")
        if any('reasoning' in param for param in model.get('supported_parameters', [])):
            tags.append("reasoning")
        if model.get('id', '').endswith(':free') or all(v == '0' for v in model.get('pricing', {}).values()):
            tags.append("free")

        # Extract only critical pricing info
        pricing = {k: v for k, v in model.get('pricing', {}).items()
                   if k in ['prompt', 'completion', 'image']}

        # Compress description - first sentence or limited chars
        description = model.get('description', '')
        first_sentence_end = description.find('.') + 1
        if 0 < first_sentence_end < 150:  # Only use if first sentence is reasonably short
            short_desc = description[:first_sentence_end].strip()
        else:
            short_desc = description[:100].strip(
            ) + ("..." if len(description) > 100 else "")

        # Build compressed model object with only essential information
        compressed_model = {
            'id': model.get('id', ''),
            'name': model.get('name', ''),
            'provider': provider,
            'context_length': model.get('context_length', 0),
            'description': short_desc,
            'tags': tags,
            'pricing': pricing,
            'modality': arch.get('modality', 'text->text')
        }

        compressed_models.append(compressed_model)

    # Add metadata about compression
    compressed_size = len(str({'models': compressed_models}))
    compression_ratio = round(
        (original_size - compressed_size) / original_size * 100, 1)

    result = {
        'models': compressed_models,
        'metadata': {
            'count': len(compressed_models),
            'original_size': original_size,
            'compressed_size': compressed_size,
            'compression_ratio': f"{compression_ratio}%"
        }
    }

    return result


def llm_picker(models_desc: str) -> list:
    """
    Given a natural language description of models, return matching models
    in the form of strings: <provider>/<model_name>.
    """
    # Load models from OpenRouter API
    url = "https://openrouter.ai/api/v1/models"
    try:
        response = requests.get(url)
        response.raise_for_status()
        models_data = response.json()
    except Exception as e:
        print(f"Error fetching models: {e}")
        return []

    # Compress the models data to reduce token usage
    compressed_data = compress_openrouter_models(models_data)

    # Print compression stats if desired
    # print(f"Compression: {compressed_data['metadata']['compression_ratio']} reduction")

    # Call LLM with compressed context
    try:
        results = ask(
            model="x-ai/grok-3-mini-beta",
            sp="You are a helpful assistant. Select models matching the user's description. Return only a list of model IDs in the format '<provider>/<model_name>', one per line.",
            prompt=f"User's model requirements: {models_desc}\n\n" +
                  f"Available models with their capabilities: {compressed_data['models']}\n\n" +
                  "Return only the matching model IDs, one per line, without any explanation or additional text."
        )
        return [line.strip() for line in results.split("\n") if line.strip()]
    except Exception as e:
        print(f"Error during LLM selection: {e}")
        return []
