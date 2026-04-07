import express from 'express';
import cors from 'cors';
import * as dotenv from 'dotenv';
dotenv.config();

import { Pinecone } from '@pinecone-database/pinecone';

const app = express();
app.use(cors());
app.use(express.json());

// Utility for pausing
const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

// Custom Fetch-based Voyage Embeddings
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
        return data.data.map(item => item.embedding);
    }

    async embedQuery(text) {
        const result = await this.embedDocuments([text]);
        return result[0];
    }
}

const OLLAMA_API_URL = "https://api.groq.com/openai/v1/chat/completions";
const OLLAMA_MODEL_NAME = "llama-3.1-8b-instant";

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

// async function callOllama(model, contents, systemInstruction, retryCount = 0) {
//     const apiKey = process.env.OLLAMA_API_KEY;
//     if (!apiKey) {
//         throw new Error('Missing OLLAMA_API_KEY. Set it in your environment or .env file.');
//     }

//     const prompt = buildOllamaPrompt(contents, systemInstruction);
//     const response = await fetch(OLLAMA_API_URL, {
//         method: 'POST',
//         headers: {
//             'Content-Type': 'application/json',
//             'Authorization': `Bearer ${apiKey}`
//         },
//         body: JSON.stringify({
//             model,
//             input: prompt,
//             options: { temperature: 0.2 }
//         })
//     });

//     const raw = await response.text();
//     let data;
//     try {
//         data = JSON.parse(raw);
//     } catch {
//         data = raw;
//     }

//     if (!response.ok) {
//         console.error('Ollama Error:', raw);
//         throw new Error(`Ollama API Error: ${response.status} ${response.statusText}: ${raw}`);
//     }

//     const text = typeof data === 'string'
//         ? data
//         : Array.isArray(data.output)
//             ? data.output.join('')
//             : typeof data.output === 'string'
//                 ? data.output
//                 : data.choices?.[0]?.message?.content || data.result || '';

//     return { text: String(text).trim() };
// }

async function callOllama(model, contents, systemInstruction) {
    const apiKey = process.env.GROQ_API_KEY;

    if (!apiKey) {
        throw new Error("Missing GROQ_API_KEY in .env");
    }

    const messages = [
        { role: "system", content: systemInstruction },
        ...contents.map(item => ({
            role: item.role === "model" ? "assistant" : item.role,
            content: Array.isArray(item.parts)
                ? item.parts.map(p => p.text).join(" ")
                : item.text || ""
        }))
    ];

    const response = await fetch(OLLAMA_API_URL, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            "Authorization": `Bearer ${apiKey}`
        },
        body: JSON.stringify({
            model: model,
            messages: messages,
            temperature: 0.2
        })
    });

    const data = await response.json();

    if (!response.ok) {
        console.error("Ollama Cloud Error:", data);
        throw new Error("Ollama Cloud API Error");
    }

    return {
        text: data.choices?.[0]?.message?.content || ""
    };
}

app.post('/api/chat', async (req, res) => {
    const { message, history } = req.body;

    try {
        // STEP 1: Search Pinecone using the RAW message to save API calls
        const embeddings = new CustomVoyageEmbeddings({
            apiKey: process.env.VOYAGEAI_API_KEY,
            modelName: 'voyage-2',
            inputType: 'query',
        });

        const queryVector = await embeddings.embedQuery(message);
        const pinecone = new Pinecone();
        const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME);

        const searchResults = await pineconeIndex.query({
            topK: 6,
            vector: queryVector,
            includeMetadata: true,
        });

        const products = searchResults.matches.map(match => ({
            id: match.id,
            name: match.metadata.name,
            price: match.metadata.price,
            link: match.metadata.link,
            category: match.metadata.category,
            text: match.metadata.text
        }));

        const context = products.map(p => p.text).join("\n\n---\n\n");

        // STEP 2: One single call to Gemini for both thinking and responding
        const updatedHistory = [...(history || []), { role: 'user', parts: [{ text: message }] }];

        const systemInstruction = `You are a LUXORA Interior Designer.

Your goal is to first understand the user's needs by asking clarifying questions, and only after receiving enough information, provide styling advice using products from the context catalog.

CONTEXT CATALOG:
${context}

RULES:

1. If user requirements are unclear, ask 1–2 clarifying questions ONLY.
2. Do NOT provide any product recommendations while asking questions.
3. Once the user answers the questions, then provide recommendations.
4. When providing recommendations, use bullet points.
5. Max 3 recommendations.
6. If the user refers to past chat items, prioritize those.
7. Keep the tone minimalist, helpful, and design-focused.
8. If the user writes in Hinglish (Hindi + English mix), respond in Hinglish. Otherwise, respond only in English.
`

        const ollamaResponse = await callOllama(OLLAMA_MODEL_NAME, updatedHistory, systemInstruction);

        const text = ollamaResponse.text.toLowerCase();

        const isRecommendation =
            text.includes("recommend") ||
            text.includes("here are") ||
            text.includes("options") ||
            text.includes("suggest") ||
            text.includes("you can consider");

        res.json({
            response: ollamaResponse.text,
            products: isRecommendation ? products.slice(0, 4) : []
        });
    } catch (err) {
        console.error(err);
        res.status(500).json({ error: err.message });
    }
});

app.get('/api/health', (req, res) => {
    res.json({ status: 'ok' });
});

const PORT = process.env.PORT || 5006;
app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
});
// You are a LUXORA Interior Designer.
// Your goal is to first understand user needs and wants by asking cross-questions, then provide crisp styling advice using products from the context catalog.

// CONTEXT CATALOG:
// ${context}

// RULES:

// 1. Ask cross-questions related to the user's needs and preferences before giving recommendations.
// 2. Use bullet points for recommendations.
// 3. Max 3 recommendations.
// 4. If the user refers to past chat items, prioritize those.
// 5. Keep the tone minimalist, helpful, and design-focused.
// 6. If the user writes in Hinglish (Hindi + English mix), respond in Hinglish. Otherwise, respond only in English.`
