# Ollama Plugin

This plugin provides integration with [Ollama](https://ollama.com/)'s local models through the ElizaOS platform. It allows you to leverage locally running LLMs for text generation, embeddings, object generation, and **multimodal vision capabilities** with image analysis.

## Overview

Ollama enables running large language models locally on your machine. This plugin connects ElizaOS with your local Ollama installation, giving your characters access to powerful language models running on your own hardware.

## Requirements

- [Ollama](https://ollama.com/) installed and running on your system
- ElizaOS platform
- At least one Ollama model pulled and available (e.g., `llama3`, `gemma3:latest`)
- For image analysis: Vision-capable models (e.g., `llava`, `llama3.2-vision`)

## Installation

1. Install this plugin in your ElizaOS project:
   ```bash
   bun add @elizaos/plugin-ollama
   ```

2. Make sure Ollama is running:
   ```bash
   ollama serve
   ```

## Usage

Add the plugin to your character configuration:

```json
"plugins": ["@elizaos-plugins/plugin-ollama"]
```

## Configuration

The plugin requires these environment variables (can be set in .env file or character settings):

```json
"settings": {
  "OLLAMA_API_ENDPOINT": "http://localhost:11434/api",
  "OLLAMA_SMALL_MODEL": "gemma3:latest",
  "OLLAMA_MEDIUM_MODEL": "gemma3:latest",
  "OLLAMA_LARGE_MODEL": "gemma3:latest",
  "OLLAMA_EMBEDDING_MODEL": "nomic-embed-text:latest"
}
```

Or in `.env` file:

```
OLLAMA_API_ENDPOINT=http://localhost:11434/api
OLLAMA_SMALL_MODEL=gemma3:latest
OLLAMA_MEDIUM_MODEL=gemma3:latest
OLLAMA_LARGE_MODEL=gemma3:latest
OLLAMA_EMBEDDING_MODEL=nomic-embed-text:latest
```

### Configuration Options

- `OLLAMA_API_ENDPOINT`: Ollama API endpoint (default: http://localhost:11434/api)
- `OLLAMA_SMALL_MODEL`: Model for simpler tasks (default: gemma3:latest)
- `OLLAMA_MEDIUM_MODEL`: Medium-complexity model (default: gemma3:latest)
- `OLLAMA_LARGE_MODEL`: Model for complex tasks (default: gemma3:latest)
- `OLLAMA_EMBEDDING_MODEL`: Model for text embeddings (default: nomic-embed-text:latest)

#### Vision Model Support

For image analysis capabilities, use vision-enabled models:

- **Recommended vision models**: `llava`, `llava-llama3`, `llava-phi3`, `llama3.2-vision`
- **Example configuration**:
  ```json
  {
    "OLLAMA_SMALL_MODEL": "llava:latest",
    "OLLAMA_LARGE_MODEL": "llama3.2-vision:latest"
  }
  ```

The plugin provides these model classes:

- `TEXT_SMALL`: Optimized for fast responses with simpler prompts
- `TEXT_LARGE`: For complex tasks requiring deeper reasoning
- `TEXT_EMBEDDING`: Text embedding model
- `OBJECT_SMALL`: JSON object generation with simpler models
- `OBJECT_LARGE`: JSON object generation with more complex models

## API Reference

For detailed information about the Ollama API used by this plugin, refer to the [official Ollama API documentation](https://github.com/ollama/ollama/blob/main/docs/api.md).



## Features

### Text Generation (Small Model)

Generate text using smaller, faster models optimized for quick responses:

```js
const text = await runtime.useModel(ModelType.TEXT_SMALL, {
  prompt: 'What is the nature of reality?',
  stopSequences: [], // optional
});
```

#### With Image Attachments

The plugin now supports multimodal text generation with vision-capable models (like `llava`):

```js
const text = await runtime.useModel(ModelType.TEXT_SMALL, {
  prompt: 'Describe what you see in this image',
  content: {
    text: 'Describe what you see in this image',
    attachments: [
      {
        id: 'image1',
        url: 'https://example.com/image.jpg',
        contentType: 'image',
        title: 'Photo to analyze'
      }
    ]
  }
});
```

### Text Generation (Large Model)

Generate comprehensive text responses using more powerful models for complex tasks:

```js
const text = await runtime.useModel(ModelType.TEXT_LARGE, {
  prompt: 'Write a detailed explanation of quantum physics',
  stopSequences: [], // optional
  maxTokens: 8192, // optional (default: 8192)
  temperature: 0.7, // optional (default: 0.7)
  frequencyPenalty: 0.7, // optional (default: 0.7)
  presencePenalty: 0.7, // optional (default: 0.7)
});
```

#### With Image Attachments

Large models also support vision capabilities when using compatible models:

```js
const text = await runtime.useModel(ModelType.TEXT_LARGE, {
  prompt: 'Analyze these images and provide detailed insights',
  content: {
    text: 'Please analyze these images',
    attachments: [
      {
        id: 'chart1',
        url: 'data:image/jpeg;base64,/9j/4AAQ...',
        contentType: 'image',
        title: 'Sales Chart'
      }
    ]
  },
  maxTokens: 4096,
  temperature: 0.3
});
```

### Text Embeddings

Generate vector embeddings for text, which can be used for semantic search or other vector operations:

```js
const embedding = await runtime.useModel(ModelType.TEXT_EMBEDDING, {
  text: 'Text to embed',
});
// or
const embedding = await runtime.useModel(ModelType.TEXT_EMBEDDING, 'Text to embed');
```

### Object Generation (Small Model)

Generate structured JSON objects using faster models:

```js
const object = await runtime.useModel(ModelType.OBJECT_SMALL, {
  prompt: 'Generate a JSON object representing a user profile',
  temperature: 0.7, // optional
});
```

### Object Generation (Large Model)

Generate complex, detailed JSON objects using more powerful models:

```js
const object = await runtime.useModel(ModelType.OBJECT_LARGE, {
  prompt: 'Generate a detailed JSON object representing a restaurant',
  temperature: 0.7, // optional
});
```

## Troubleshooting

### Connection Issues

- Ensure Ollama is running by testing the `OLLAMA_API_ENDPOINT`
- Verify the `OLLAMA_API_ENDPOINT` points to the correct host and port
- Check firewall settings if connecting to a remote Ollama instance

### Model Availability

- The plugin will attempt to download models automatically if they're not found
- You can pre-download models using `ollama pull modelname`
- Check model availability with `ollama list`

### Image Support Issues

- **Vision capabilities require compatible models**: Use `llava`, `llama3.2-vision`, `bakllava`, or other vision-enabled models
- **Download vision models**: `ollama pull llava` or `ollama pull llama3.2-vision:latest`
- **Image format support**: The plugin supports images via URLs or base64 data URIs
- **Network access**: If using image URLs, ensure Ollama can access the image sources
- **Model context**: Vision models may require more memory and processing time


## License

See LICENSE file for details.
