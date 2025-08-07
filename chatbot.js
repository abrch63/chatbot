import { pipeline } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.5.1';

let chatbot = await pipeline('text-generation', 'Xenova/phi-2', { quantized: true });

let chatDiv = document.getElementById('chat');
let inputBox = document.getElementById('input');

let data = await fetch('data.json').then(res => res.json());

// === Utility: Cosine Similarity ===
function cosineSimilarity(vecA, vecB) {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < vecA.length; i++) {
    dot += vecA[i] * vecB[i];
    normA += vecA[i] * vecA[i];
    normB += vecB[i] * vecB[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

// === Get top matching chunk ===
async function findRelevantContext(question) {
  const encoder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
  let qVec = await encoder(question, { pooling: 'mean', normalize: true });
  let similarities = data.map(d => cosineSimilarity(qVec.data, d.vector));
  let topIndex = similarities.indexOf(Math.max(...similarities));
  return data[topIndex].text;
}

// === Handle user message ===
inputBox.addEventListener('keypress', async (e) => {
  if (e.key === 'Enter') {
    let question = inputBox.value.trim();
    if (!question) return;

    chatDiv.innerHTML += `<div><b>You:</b> ${question}</div>`;
    inputBox.value = 'Thinking...';
    inputBox.disabled = true;

    let context = await findRelevantContext(question);

    let prompt = `You are a helpful assistant for Vision Academy.\nContext: ${context}\nQuestion: ${question}\nAnswer:`;

    let result = await chatbot(prompt, {
      max_new_tokens: 100,
      temperature: 0.7,
    });

    let answer = result[0].generated_text.split('Answer:')[1] || result[0].generated_text;

    chatDiv.innerHTML += `<div><b>Bot:</b> ${answer.trim()}</div>`;
    inputBox.value = '';
    inputBox.disabled = false;
    chatDiv.scrollTop = chatDiv.scrollHeight;
  }
});
