import * as dotenv from 'dotenv';
dotenv.config();
import readlineSync from 'readline-sync';

// Utility for pausing
const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

// Custom Fetch-based Voyage Embeddings (Keeping your reliable version)
class CustomVoyageEmbeddings {
    constructor(config) {
        this.apiKey = config.apiKey;
        this.modelName = config.modelName || 'voyage-2';
        this.inputType = config.inputType || 'query';
    }

    async embedDocuments(texts) {
        const response = await fetch('https://api.voyageai.com/v1/embeddings', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${this.apiKey}`
            },
            body: JSON.stringify({
                input: texts,
                model: this.modelName,
                input_type: this.inputType
            })
        });

        const data = await response.json();
        if (!response.ok) {
            console.error("Voyage AI Error:", data);
            throw new Error(`Voyage AI API Error: ${response.statusText}`);
        }

        if (!data.data || !Array.isArray(data.data)) {
            console.error("Unexpected Voyage AI Response:", data);
            throw new Error("Invalid response format from Voyage AI");
        }

        return data.data.map(item => item.embedding);
    }

    async embedQuery(text) {
        const result = await this.embedDocuments([text]);
        return result[0];
    }
}

import { Pinecone } from '@pinecone-database/pinecone';

const OLLAMA_API_URL = process.env.OLLAMA_API_URL || 'https://api.ollama.com/v1/predict';
const OLLAMA_MODEL_NAME = process.env.OLLAMA_MODEL_NAME || 'summarizer';
const History = []

function buildOllamaPrompt(contents, systemInstruction) {
    const historyText = Array.isArray(contents)
        ? contents.map(item => {
            const role = item.role || 'user';
            const text = Array.isArray(item.parts)
                ? item.parts.map(part => part?.text || String(part || '')).join(' ')
                : String(item.text || '');
            return `${role.toUpperCase()}: ${text}`;
        }).join('\n')
        : String(contents || '');

    return `${systemInstruction}\n\n${historyText}`.trim();
}

async function callOllama(model, contents, systemInstruction, retryCount = 0) {
    const apiKey = process.env.OLLAMA_API_KEY;
    if (!apiKey) {
        throw new Error('Missing OLLAMA_API_KEY. Set it in your environment or .env file.');
    }

    try {
        const prompt = buildOllamaPrompt(contents, systemInstruction);
        const response = await fetch(OLLAMA_API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${apiKey}`
            },
            body: JSON.stringify({
                model,
                input: prompt,
                options: { temperature: 0.2 }
            })
        });

        const data = await response.json();
        if (!response.ok) {
            console.error('Ollama Error:', data);
            throw new Error(`Ollama API Error: ${response.statusText}`);
        }

        const text = Array.isArray(data.output)
            ? data.output.join('')
            : typeof data.output === 'string'
                ? data.output
                : data.choices?.[0]?.message?.content || data.result || '';

        return { text: text.trim() };
    } catch (error) {
        if (error.status === 429 && retryCount < 3) {
            console.log(`\n[OLLAMA Busy] Rate limit hit. Waiting 10s before retry ${retryCount + 1}...`);
            await sleep(10000);
            return callOllama(model, contents, systemInstruction, retryCount + 1);
        }
        throw error;
    }
}

async function transformQuery(question) {
    // Optimization: Skip rewriting for the very first message
    if (History.length === 0) {
        return question;
    }

    History.push({
        role: 'user',
        parts: [{ text: question }]
    })

    const systemInstruction = `You are a query rewriting expert. Based on the provided chat history, rephrase the "Follow Up user Question" into a complete, standalone question that can be understood without the chat history. Only output the rewritten question and nothing else.`;

    const response = await callOllama(OLLAMA_MODEL_NAME, History, systemInstruction);

    History.pop()

    return response.text
}

async function chatting(question) {
    try {
        const queries = await transformQuery(question);

        const embeddings = new CustomVoyageEmbeddings({
            apiKey: process.env.VOYAGEAI_API_KEY,
            modelName: 'voyage-2',
            inputType: 'query',
        });

        const queryVector = await embeddings.embedQuery(queries);

        const pinecone = new Pinecone();
        const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME);

        const searchResults = await pineconeIndex.query({
            topK: 10,
            vector: queryVector,
            includeMetadata: true,
        });

        const context = searchResults.matches
            .map(match => match.metadata.text)
            .join("\n\n---\n\n");

        History.push({
            role: 'user',
            parts: [{ text: queries }]
        })

        const systemInstruction = `You are an IKEA-style Interior Designer and Home Decor Expert.
Your goal is to help users find the perfect furniture for their space while providing styling advice.

GUIDELINES:
1. KEEP IT CRISP AND CONCISE. Avoid long paragraphs.
2. Max 3-4 product recommendations per response.
3. Use bullet points for readability.
4. ONLY use products found in the provided context catalog.
5. For every recommendation, suggest ONE related item for upselling.
6. Mention the price (e.g., "$249") and provide the link.

Context Catalog: ${context}`;

        const response = await callOllama(OLLAMA_MODEL_NAME, History, systemInstruction);

        History.push({
            role: 'model',
            parts: [{ text: response.text }]
        })

        console.log("\n[Designer]:", response.text);
    } catch (err) {
        console.error("\n[Error]:", err.message);
    }
}

async function main() {
    console.log("\n--- IKEA Interior Design Bot (Ready) ---");
    const userProblem = readlineSync.question("\nAsk me anything--> ");

    if (userProblem.toLowerCase() === 'exit' || userProblem.toLowerCase() === 'quit') {
        console.log("Goodbye!");
        process.exit();
    }

    await chatting(userProblem);
    main();
}

main();