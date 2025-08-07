import { pipeline } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.5.1';

// Load a tiny model that 100% works with no token
let chatbot = await pipeline('text-generation', 'Xenova/tiny-random-GPT2', { quantized: true });

// Your encoder (for future embedding similarity)
let encoder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
