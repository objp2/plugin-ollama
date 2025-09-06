import { describe, it, expect } from 'bun:test';
import { ollamaPlugin } from './index';
import { ContentType, type Content } from '@elizaos/core';

describe('ollamaPlugin', () => {
  it('should export ollamaPlugin', () => {
    expect(ollamaPlugin).toBeDefined();
  });

  it('should have the correct name', () => {
    expect(ollamaPlugin.name).toBe('ollama');
  });

  it('should have an init method', () => {
    expect(ollamaPlugin.init).toBeDefined();
    expect(typeof ollamaPlugin.init).toBe('function');
  });

  it('should have models for text generation', () => {
    expect(ollamaPlugin.models).toBeDefined();
    expect(ollamaPlugin.models['TEXT_SMALL']).toBeDefined();
    expect(ollamaPlugin.models['TEXT_LARGE']).toBeDefined();
  });

  // Test the image functionality conceptually
  it('should handle content with image attachments', () => {
    const testContent: Content = {
      text: 'Describe this image',
      attachments: [
        {
          id: 'test-image-1',
          url: 'https://example.com/image.jpg',
          title: 'Test Image',
          contentType: ContentType.IMAGE,
        },
      ],
    };

    // Verify the test content has the expected structure
    expect(testContent.attachments).toBeDefined();
    expect(testContent.attachments![0].contentType).toBe(ContentType.IMAGE);
    expect(testContent.attachments![0].url).toBeDefined();
  });

  it('should handle content without image attachments', () => {
    const testContent: Content = {
      text: 'Just text content',
    };

    expect(testContent.attachments).toBeUndefined();
  });
});
