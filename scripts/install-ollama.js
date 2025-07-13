#!/usr/bin/env bun

import { spawnSync } from 'bun';
import { platform } from 'os';

const isWindows = platform() === 'win32';
const isMac = platform() === 'darwin';
const isLinux = platform() === 'linux';

console.log('Installing Ollama...');

try {
  // Check if Ollama is already installed
  const checkResult = spawnSync(['ollama', '--version'], {
    stdout: 'pipe',
    stderr: 'pipe'
  });
  
  if (checkResult.exitCode === 0) {
    console.log('Ollama is already installed.');
  } else {
    // Ollama not installed, proceed with installation
    if (isWindows) {
      console.log('Please download and install Ollama from: https://ollama.com/download/windows');
      console.log('After installation, run: ollama pull nomic-embed-text');
      process.exit(0);
    } else if (isMac || isLinux) {
      console.log('Installing Ollama using the official installer...');
      const installResult = spawnSync(['sh', '-c', 'curl -fsSL https://ollama.com/install.sh | sh'], {
        stdout: 'inherit',
        stderr: 'inherit'
      });
      
      if (installResult.exitCode !== 0) {
        throw new Error('Failed to install Ollama');
      }
    } else {
      console.error('Unsupported platform. Please install Ollama manually from: https://ollama.com/download');
      process.exit(1);
    }
  }

  // Pull the required embedding model
  console.log('\nPulling required embedding model...');
  console.log('Pulling nomic-embed-text...');
  const pullNomic = spawnSync(['ollama', 'pull', 'nomic-embed-text'], {
    stdout: 'inherit',
    stderr: 'inherit'
  });
  
  if (pullNomic.exitCode !== 0) {
    throw new Error('Failed to pull nomic-embed-text model');
  }
  
  console.log('\nOllama installation and model setup complete!');
} catch (error) {
  console.error('Error during installation:', error.message);
  console.error('\nPlease install Ollama manually from: https://ollama.com/download');
  process.exit(1);
}