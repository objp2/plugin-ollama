import type { ObjectGenerationParams, Plugin, TextEmbeddingParams } from '@elizaos/core';
import { type GenerateTextParams, ModelType, logger, ContentType, type Content } from '@elizaos/core';
import { generateObject, generateText, embed } from 'ai';
import { createOllama } from 'ollama-ai-provider';

// Default Ollama API URL
const OLLAMA_API_URL = 'http://localhost:11434/api';

/**
 * Fetches an image from URL and converts it to base64 format for Ollama API
 */
async function imageUrlToBase64(url: string, fetch?: typeof globalThis.fetch): Promise<string | null> {
  try {
    const fetchFn = fetch || globalThis.fetch;
    const response = await fetchFn(url);
    if (!response.ok) {
      logger.warn(`Failed to fetch image from ${url}: ${response.statusText}`);
      return null;
    }
    
    const arrayBuffer = await response.arrayBuffer();
    const base64String = Buffer.from(arrayBuffer).toString('base64');
    
    // Get content type from response or default to generic image
    const contentType = response.headers.get('content-type') || 'image/jpeg';
    
    return `data:${contentType};base64,${base64String}`;
  } catch (error) {
    logger.error({ error, url }, 'Error converting image URL to base64');
    return null;
  }
}

/**
 * Safely stringify objects that may contain circular references or non-serializable properties
 */
function safeStringify(obj: any, maxDepth = 3): string {
  const seen = new WeakSet();
  
  function replacer(key: string, value: any, depth = 0): any {
    if (depth > maxDepth) {
      return '[Max Depth Reached]';
    }
    
    if (value === null) return null;
    if (typeof value === 'undefined') return '[Undefined]';
    if (typeof value === 'function') return '[Function]';
    if (typeof value === 'symbol') return '[Symbol]';
    if (value instanceof Error) return { name: value.name, message: value.message, stack: value.stack };
    
    if (typeof value === 'object') {
      if (seen.has(value)) {
        return '[Circular Reference]';
      }
      seen.add(value);
      
      if (Array.isArray(value)) {
        return value.map((item, index) => replacer(String(index), item, depth + 1));
      }
      
      const result: any = {};
      for (const [k, v] of Object.entries(value)) {
        if (k === 'fetch' || k === 'runtime') {
          result[k] = '[Hidden]';
        } else {
          result[k] = replacer(k, v, depth + 1);
        }
      }
      return result;
    }
    
    return value;
  }
  
  try {
    return JSON.stringify(replacer('', obj), null, 2);
  } catch (error) {
    return `[Stringify Error: ${error.message}]`;
  }
}

/**
 * Extracts images from ElizaOS content and converts them to base64 format
 */
async function extractImagesFromContent(content: Content, fetch?: typeof globalThis.fetch): Promise<string[]> {
  const images: string[] = [];
  
  logger.log(`[Ollama] extractImagesFromContent called with content: ${safeStringify(content)}`);
  
  if (!content.attachments) {
    logger.log(`[Ollama] No attachments found in content`);
    return images;
  }
  
  logger.log(`[Ollama] Found ${content.attachments.length} attachments`);
  
  for (const attachment of content.attachments) {
    logger.log(`[Ollama] Processing attachment: ${safeStringify(attachment)}`);
    logger.log(`[Ollama] Attachment contentType: "${attachment.contentType}", expected: "${ContentType.IMAGE}"`);
    logger.log(`[Ollama] ContentType.IMAGE value: "${ContentType.IMAGE}"`);
    logger.log(`[Ollama] Type of contentType: ${typeof attachment.contentType}`);
    logger.log(`[Ollama] Strict equality: ${attachment.contentType === ContentType.IMAGE}`);
    logger.log(`[Ollama] Loose equality: ${attachment.contentType == ContentType.IMAGE}`);
    logger.log(`[Ollama] Lowercase comparison: ${attachment.contentType?.toLowerCase() === 'image'}`);
    
    // Try multiple comparison methods to debug the issue
    const isImageType = attachment.contentType === ContentType.IMAGE || 
                        attachment.contentType?.toLowerCase() === 'image';
    
    if (isImageType && attachment.url) {
      logger.log(`[Ollama] Processing image attachment: ${attachment.title || attachment.url}`);
      const base64Image = await imageUrlToBase64(attachment.url, fetch);
      if (base64Image) {
        images.push(base64Image);
        logger.log(`[Ollama] Added image attachment: ${attachment.title || attachment.url}`);
      } else {
        logger.warn(`[Ollama] Failed to convert image to base64: ${attachment.url}`);
      }
    } else {
      logger.log(`[Ollama] Skipping non-image attachment or attachment without URL. ContentType: "${attachment.contentType}", URL: ${attachment.url}`);
    }
  }
  
  logger.log(`[Ollama] Total images extracted: ${images.length}`);
  return images;
}

/**
 * Retrieves the Ollama API base URL from runtime settings.
 *
 * If the API endpoint is not set in the runtime, defaults to the standard Ollama URL.
 * The URL should include the /api path for ollama-ai-provider compatibility.
 *
 * @returns The base URL for the Ollama API.
 */
function getBaseURL(runtime: { getSetting: (key: string) => string | undefined }): string {
  const apiEndpoint =
    runtime.getSetting('OLLAMA_API_ENDPOINT') ||
    runtime.getSetting('OLLAMA_API_URL') ||
    OLLAMA_API_URL;

  // Ensure the URL ends with /api for ollama-ai-provider
  if (!apiEndpoint.endsWith('/api')) {
    return apiEndpoint.endsWith('/') ? `${apiEndpoint}api` : `${apiEndpoint}/api`;
  }
  return apiEndpoint;
}

/**
 * Ensures that the specified Ollama model is available locally, downloading it if necessary.
 *
 * Checks for the presence of the model via the Ollama API and attempts to download it if not found. Logs progress and errors during the process.
 */
async function ensureModelAvailable(
  runtime: {
    getSetting: (key: string) => string | undefined;
    fetch?: typeof fetch;
  },
  model: string,
  providedBaseURL?: string
) {
  const baseURL = providedBaseURL || getBaseURL(runtime);
  // Remove /api suffix for direct API calls
  const apiBase = baseURL.endsWith('/api') ? baseURL.slice(0, -4) : baseURL;
  try {
    const showRes = await fetch(`${apiBase}/api/show`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model }),
    });
    if (showRes.ok) return;
    logger.info(`[Ollama] Model ${model} not found locally. Downloading...`);
    const pullRes = await fetch(`${apiBase}/api/pull`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model, stream: false }),
    });
    if (!pullRes.ok) {
      logger.error(`Failed to pull model ${model}: ${pullRes.statusText}`);
    } else {
      logger.info(`[Ollama] Downloaded model ${model}`);
    }
  } catch (err) {
    logger.error({ error: err }, 'Error ensuring model availability');
  }
}

/**
 * Generates text from the Ollama API using the specified model and parameters.
 *
 * Returns the generated text, or an error message if generation fails.
 */
async function generateOllamaText(
  ollama: ReturnType<typeof createOllama>,
  model: string,
  params: {
    prompt: string;
    system?: string;
    temperature: number;
    maxTokens: number;
    frequencyPenalty: number;
    presencePenalty: number;
    stopSequences: string[];
    images?: string[];
  }
) {
  try {
    // If images are provided, use messages format for multimodal support
    if (params.images && params.images.length > 0) {
      const messages: Array<
        | { role: 'system'; content: string }
        | { role: 'user'; content: Array<{ type: 'text'; text: string } | { type: 'image'; image: string }> }
      > = [];
      
      // Add system message if provided
      if (params.system) {
        messages.push({
          role: 'system',
          content: params.system,
        });
      }
      
      // Create user message with text and images
      const userContent: Array<{ type: 'text'; text: string } | { type: 'image'; image: string }> = [
        { type: 'text', text: params.prompt }
      ];
      
      // Add images as base64 data URLs
      for (const imageData of params.images) {
        userContent.push({
          type: 'image',
          image: imageData
        });
      }
      
      messages.push({
        role: 'user',
        content: userContent,
      });

      const { text: ollamaResponse } = await generateText({
        model: ollama(model),
        messages,
        temperature: params.temperature,
        maxTokens: params.maxTokens,
        frequencyPenalty: params.frequencyPenalty,
        presencePenalty: params.presencePenalty,
        stopSequences: params.stopSequences,
      });
      return ollamaResponse;
    } else {
      // Use simple prompt format when no images
      const { text: ollamaResponse } = await generateText({
        model: ollama(model),
        prompt: params.prompt,
        system: params.system,
        temperature: params.temperature,
        maxTokens: params.maxTokens,
        frequencyPenalty: params.frequencyPenalty,
        presencePenalty: params.presencePenalty,
        stopSequences: params.stopSequences,
      });
      return ollamaResponse;
    }
  } catch (error: unknown) {
    logger.error({ error }, 'Error in generateOllamaText');
    return 'Error generating text. Please try again later.';
  }
}

/**
 * Generates an object from the Ollama API using the specified model and parameters.
 *
 * Returns the generated object, or an empty object if generation fails.
 */
async function generateOllamaObject(
  ollama: ReturnType<typeof createOllama>,
  model: string,
  params: ObjectGenerationParams
) {
  try {
    const { object } = await generateObject({
      model: ollama(model),
      output: 'no-schema',
      prompt: params.prompt,
      temperature: params.temperature,
    });
    return object;
  } catch (error: unknown) {
    logger.error({ error }, 'Error generating object');
    return {};
  }
}

export const ollamaPlugin: Plugin = {
  name: 'ollama',
  description: 'Ollama plugin',
  config: {
    OLLAMA_API_ENDPOINT: process.env.OLLAMA_API_ENDPOINT,
    OLLAMA_SMALL_MODEL: process.env.OLLAMA_SMALL_MODEL,
    OLLAMA_MEDIUM_MODEL: process.env.OLLAMA_MEDIUM_MODEL,
    OLLAMA_LARGE_MODEL: process.env.OLLAMA_LARGE_MODEL,
    OLLAMA_EMBEDDING_MODEL: process.env.OLLAMA_EMBEDDING_MODEL,
  },
  async init(_config, runtime) {
    const baseURL = getBaseURL(runtime);

    // Check if endpoint is configured
    if (!baseURL || baseURL === 'http://localhost:11434/api') {
      const endpoint = runtime.getSetting('OLLAMA_API_ENDPOINT');
      if (!endpoint) {
        logger.warn(
          'OLLAMA_API_ENDPOINT is not set in environment - Ollama functionality will use default localhost:11434'
        );
      }
    }

    try {
      // Validate Ollama API endpoint by checking if it's accessible
      // Remove /api suffix for direct API calls
      const apiBase = baseURL.endsWith('/api') ? baseURL.slice(0, -4) : baseURL;
      const response = await fetch(`${apiBase}/api/tags`, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
      });

      if (!response.ok) {
        logger.warn(`Ollama API endpoint validation failed: ${response.statusText}`);
        logger.warn('Ollama functionality will be limited until a valid endpoint is provided');
      } else {
        const data = (await response.json()) as {
          models?: Array<{ name: string }>;
        };
        const modelCount = data?.models?.length || 0;
        logger.log(
          `Ollama API endpoint validated successfully. Found ${modelCount} models available.`
        );
      }
    } catch (fetchError: unknown) {
      const message = fetchError instanceof Error ? fetchError.message : String(fetchError);
      logger.warn(`Error validating Ollama API endpoint: ${message}`);
      logger.warn(
        'Ollama functionality will be limited until a valid endpoint is provided - Make sure Ollama is running at ${baseURL}'
      );
    }
  },
  models: {
    [ModelType.TEXT_EMBEDDING]: async (
      runtime,
      params: TextEmbeddingParams | string | null
    ): Promise<number[]> => {
      try {
        const baseURL = getBaseURL(runtime);
        const ollama = createOllama({
          fetch: runtime.fetch,
          baseURL,
        });

        const modelName = runtime.getSetting('OLLAMA_EMBEDDING_MODEL') || 'nomic-embed-text:latest';
        logger.log(`[Ollama] Using TEXT_EMBEDDING model: ${modelName}`);
        await ensureModelAvailable(runtime, modelName, baseURL);
        const text =
          typeof params === 'string'
            ? params
            : params
              ? (params as TextEmbeddingParams).text || ''
              : '';

        // If no text is provided (e.g., for dimension detection), use a default text
        const embeddingText = text || 'test';

        if (!text) {
          logger.debug(
            'No text provided for embedding, using default text for dimension detection'
          );
        }

        // Use ollama.embedding() as shown in the docs
        try {
          const { embedding } = await embed({
            model: ollama.embedding(modelName),
            value: embeddingText,
          });
          return embedding;
        } catch (embeddingError) {
          logger.error({ error: embeddingError }, 'Error generating embedding');
          return Array(1536).fill(0);
        }
      } catch (error) {
        logger.error({ error }, 'Error in TEXT_EMBEDDING model');
        // Return a fallback vector rather than crashing
        return Array(1536).fill(0);
      }
    },
    [ModelType.TEXT_SMALL]: async (runtime, params: GenerateTextParams & { content?: Content; images?: string[] }) => {
      try {
        logger.log(`[Ollama] TEXT_SMALL called with params keys: [${Object.keys(params).join(', ')}]`);
        logger.log(`[Ollama] TEXT_SMALL prompt length: ${params.prompt?.length || 0}`);
        logger.log(`[Ollama] TEXT_SMALL has content: ${!!params.content}`);
        logger.log(`[Ollama] TEXT_SMALL has images: ${!!params.images}`);
        const { prompt, stopSequences = [], images: providedImages, content } = params;
        const temperature = 0.7;
        const frequency_penalty = 0.7;
        const presence_penalty = 0.7;
        const max_response_length = 8000;
        const baseURL = getBaseURL(runtime);
        const ollama = createOllama({
          fetch: runtime.fetch,
          baseURL,
        });

        const model =
          runtime.getSetting('OLLAMA_SMALL_MODEL') ||
          runtime.getSetting('SMALL_MODEL') ||
          'gemma3:latest';

        logger.log(`[Ollama] Using TEXT_SMALL model: ${model}`);
        await ensureModelAvailable(runtime, model, baseURL);
        
        // Extract images from content if available
        let images = providedImages || [];
        logger.log(`[Ollama] Content provided to TEXT_SMALL: ${JSON.stringify(content, null, 2)}`);
        logger.log(`[Ollama] Provided images to TEXT_SMALL: ${providedImages?.length || 0}`);
        
        if (content && images.length === 0) {
          logger.log(`[Ollama] Attempting to extract images from content in TEXT_SMALL...`);
          images = await extractImagesFromContent(content, runtime.fetch);
          logger.log(`[Ollama] Extracted ${images.length} images from content in TEXT_SMALL`);
        }
        
        if (images.length > 0) {
          logger.log(`[Ollama] Processing ${images.length} image(s) with text generation`);
        }
        
        logger.log('generating text');
        logger.log(prompt);

        return await generateOllamaText(ollama, model, {
          prompt,
          system: runtime.character?.system || undefined,
          temperature,
          maxTokens: max_response_length,
          frequencyPenalty: frequency_penalty,
          presencePenalty: presence_penalty,
          stopSequences,
          images,
        });
      } catch (error) {
        logger.error(`[Ollama] Error in TEXT_SMALL model: ${error?.message || 'Unknown error'}`);
        logger.error(`[Ollama] Error type: ${typeof error}`);
        logger.error(`[Ollama] Error stack: ${error?.stack || 'No stack trace'}`);
        logger.error(`[Ollama] Error details: ${safeStringify(error)}`);
        return 'Error generating text. Please try again later.';
      }
    },
    [ModelType.TEXT_LARGE]: async (
      runtime,
      params: GenerateTextParams & { content?: Content; images?: string[] }
    ) => {
      try {
        logger.log(`[Ollama] TEXT_LARGE called with params keys: [${Object.keys(params).join(', ')}]`);
        logger.log(`[Ollama] TEXT_LARGE prompt length: ${params.prompt?.length || 0}`);
        logger.log(`[Ollama] TEXT_LARGE has content: ${!!params.content}`);
        logger.log(`[Ollama] TEXT_LARGE has images: ${!!params.images}`);
        const {
          prompt,
          stopSequences = [],
          maxTokens = 8192,
          temperature = 0.7,
          frequencyPenalty = 0.7,
          presencePenalty = 0.7,
          images: providedImages,
          content
        } = params;
        
        const model =
          runtime.getSetting('OLLAMA_LARGE_MODEL') ||
          runtime.getSetting('LARGE_MODEL') ||
          'gemma3:latest';
        const baseURL = getBaseURL(runtime);
        const ollama = createOllama({
          fetch: runtime.fetch,
          baseURL,
        });

        logger.log(`[Ollama] Using TEXT_LARGE model: ${model}`);
        await ensureModelAvailable(runtime, model, baseURL);
        
        // Extract images from content if available
        let images = providedImages || [];
        logger.log(`[Ollama] Content provided to TEXT_LARGE: ${safeStringify(content)}`);
        logger.log(`[Ollama] Provided images to TEXT_LARGE: ${providedImages?.length || 0}`);
        
        if (content && images.length === 0) {
          logger.log(`[Ollama] Attempting to extract images from content in TEXT_LARGE...`);
          images = await extractImagesFromContent(content, runtime.fetch);
          logger.log(`[Ollama] Extracted ${images.length} images from content in TEXT_LARGE`);
        }
        
        if (images.length > 0) {
          logger.log(`[Ollama] Processing ${images.length} image(s) with text generation`);
        }
        
        return await generateOllamaText(ollama, model, {
          prompt,
          system: runtime.character?.system || undefined,
          temperature,
          maxTokens,
          frequencyPenalty,
          presencePenalty,
          stopSequences,
          images,
        });
      } catch (error) {
        logger.error(`[Ollama] Error in TEXT_LARGE model: ${error?.message || 'Unknown error'}`);
        logger.error(`[Ollama] Error type: ${typeof error}`);
        logger.error(`[Ollama] Error stack: ${error?.stack || 'No stack trace'}`);
        logger.error(`[Ollama] Error details: ${safeStringify(error)}`);
        return 'Error generating text. Please try again later.';
      }
    },
    [ModelType.OBJECT_SMALL]: async (runtime, params: ObjectGenerationParams) => {
      try {
        const baseURL = getBaseURL(runtime);
        const ollama = createOllama({
          fetch: runtime.fetch,
          baseURL,
        });
        const model =
          runtime.getSetting('OLLAMA_SMALL_MODEL') ||
          runtime.getSetting('SMALL_MODEL') ||
          'gemma3:latest';

        logger.log(`[Ollama] Using OBJECT_SMALL model: ${model}`);
        await ensureModelAvailable(runtime, model, baseURL);
        if (params.schema) {
          logger.info('Using OBJECT_SMALL without schema validation');
        }

        return await generateOllamaObject(ollama, model, params);
      } catch (error) {
        logger.error({ error }, 'Error in OBJECT_SMALL model');
        // Return empty object instead of crashing
        return {};
      }
    },
    [ModelType.OBJECT_LARGE]: async (runtime, params: ObjectGenerationParams) => {
      try {
        const baseURL = getBaseURL(runtime);
        const ollama = createOllama({
          fetch: runtime.fetch,
          baseURL,
        });
        const model =
          runtime.getSetting('OLLAMA_LARGE_MODEL') ||
          runtime.getSetting('LARGE_MODEL') ||
          'gemma3:latest';

        logger.log(`[Ollama] Using OBJECT_LARGE model: ${model}`);
        await ensureModelAvailable(runtime, model, baseURL);
        if (params.schema) {
          logger.info('Using OBJECT_LARGE without schema validation');
        }

        return await generateOllamaObject(ollama, model, params);
      } catch (error) {
        logger.error({ error }, 'Error in OBJECT_LARGE model');
        // Return empty object instead of crashing
        return {};
      }
    },
  },
  tests: [
    {
      name: 'ollama_plugin_tests',
      tests: [
        {
          name: 'ollama_test_url_validation',
          fn: async (runtime) => {
            try {
              const baseURL = getBaseURL(runtime);
              // Remove /api suffix for direct API calls
              const apiBase = baseURL.endsWith('/api') ? baseURL.slice(0, -4) : baseURL;
              const response = await fetch(`${apiBase}/api/tags`);
              const data = await response.json();
              const modelCount =
                data && typeof data === 'object' && 'models' in data && Array.isArray(data.models)
                  ? data.models.length
                  : 0;
              logger.log(`Models Available: ${modelCount}`);
              if (!response.ok) {
                logger.error(`Failed to validate Ollama API: ${response.statusText}`);
                return;
              }
            } catch (error) {
              logger.error({ error }, 'Error in ollama_test_url_validation');
            }
          },
        },
        {
          name: 'ollama_test_text_embedding',
          fn: async (runtime) => {
            try {
              const embedding = await runtime.useModel(ModelType.TEXT_EMBEDDING, {
                text: 'Hello, world!',
              });
              logger.log({ embedding }, 'Generated embedding');
            } catch (error) {
              logger.error({ error }, 'Error in test_text_embedding');
            }
          },
        },
        {
          name: 'ollama_test_text_large',
          fn: async (runtime) => {
            try {
              const text = await runtime.useModel(ModelType.TEXT_LARGE, {
                prompt: 'What is the nature of reality in 10 words?',
              });
              if (text.length === 0) {
                logger.error('Failed to generate text');
                return;
              }
              logger.log({ text }, 'Generated with test_text_large');
            } catch (error) {
              logger.error({ error }, 'Error in test_text_large');
            }
          },
        },
        {
          name: 'ollama_test_text_small',
          fn: async (runtime) => {
            try {
              const text = await runtime.useModel(ModelType.TEXT_SMALL, {
                prompt: 'What is the nature of reality in 10 words?',
              });
              if (text.length === 0) {
                logger.error('Failed to generate text');
                return;
              }
              logger.log({ text }, 'Generated with test_text_small');
            } catch (error) {
              logger.error({ error }, 'Error in test_text_small');
            }
          },
        },
        {
          name: 'ollama_test_text_with_images',
          fn: async (runtime) => {
            try {
              // Test with content that includes images
              const textParams = {
                prompt: 'Describe this image',
                content: {
                  text: 'Describe this image',
                  attachments: [
                    {
                      id: 'test-image',
                      // Small 1x1 pixel test image in base64
                      url: 'data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/2wBDAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwA/wA==',
                      title: 'Test Image',
                      contentType: ContentType.IMAGE,
                    },
                  ],
                },
              };
              
              const text = await runtime.useModel(ModelType.TEXT_SMALL, textParams);
              if (text.length === 0) {
                logger.error('Failed to generate text with images');
                return;
              }
              logger.log({ text }, 'Generated text with image content');
            } catch (error) {
              // This test may fail if the model doesn't support vision
              logger.warn({ error }, 'Image test failed - may need vision-capable model like llava');
            }
          },
        },
        {
          name: 'ollama_test_object_small',
          fn: async (runtime) => {
            try {
              const object = await runtime.useModel(ModelType.OBJECT_SMALL, {
                prompt:
                  'Generate a JSON object representing a user profile with name, age, and hobbies',
                temperature: 0.7,
                schema: undefined,
              });
              logger.log({ object }, 'Generated object');
            } catch (error) {
              logger.error({ error }, 'Error in test_object_small');
            }
          },
        },
        {
          name: 'ollama_test_object_large',
          fn: async (runtime) => {
            try {
              const object = await runtime.useModel(ModelType.OBJECT_LARGE, {
                prompt:
                  'Generate a detailed JSON object representing a restaurant with name, cuisine type, menu items with prices, and customer reviews',
                temperature: 0.7,
                schema: undefined,
              });
              logger.log({ object }, 'Generated object');
            } catch (error) {
              logger.error({ error }, 'Error in test_object_large');
            }
          },
        },
      ],
    },
  ],
};
export default ollamaPlugin;
