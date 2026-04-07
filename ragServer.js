// PDF load karne ka index.js file
import * as dotenv from 'dotenv';
dotenv.config();

import { Document } from '@langchain/core/documents';
import { readFileSync } from 'fs';
import { parse } from 'csv-parse/sync';
// We are bypassing the buggy @langchain/community Voyage class and using fetch directly
class CustomVoyageEmbeddings {
    constructor(config) {
        this.apiKey = config.apiKey;
        this.modelName = config.modelName || 'voyage-2';
        this.inputType = config.inputType || 'document';
    }

    async embedDocuments(texts) {
        if (!texts || texts.length === 0) {
            return [];
        }
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
import { PineconeStore } from '@langchain/pinecone';


async function indexDocument() {

    // Load IKEA CSV Catalog
    const csvPath = './data/ikeaSaFurniture.csv';
    const fileContent = readFileSync(csvPath, 'utf8');

    // Parse CSV
    const records = parse(fileContent, {
        columns: true,
        skip_empty_lines: true,
        trim: true
    });

    console.log(`Loaded ${records.length} products from CSV`);

    // Create Documents from CSV records
    const chunkedDocs = records.map(record => {
        // Data Validation: Ensure we don't send empty or invalid data
        const name = record.name || "Unnamed Product";
        const category = record.category || "General";
        const price = record.price || "Contact for Price";
        const description = record.short_description || "No description available.";
        
        const content = `Product: ${name}
Category: ${category}
Price: ${price}
Description: ${description}
Designer: ${record.designer || "N/A"}
Dimensions: ${record.depth || "?"}x${record.height || "?"}x${record.width || "?"}
Link: ${record.link || "#"}`;

        return new Document({
            pageContent: content,
            metadata: {
                id: record.item_id || Math.random().toString(36).substr(2, 9),
                name: name,
                price: price,
                link: record.link || "#",
                category: category
            }
        });
    });

    console.log("Documents ready for embedding");

    // vector Embedding model
    const embeddings = new CustomVoyageEmbeddings({
        apiKey: process.env.VOYAGEAI_API_KEY,
        modelName: 'voyage-2',
        inputType: 'document',
    });

    console.log("Embedding model configured")


    //   Database ko bhi configure
    //  Initialize Pinecone Client

    const pinecone = new Pinecone();
    const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME);
    console.log("Pinecone configured")

    // Clear existing index to avoid duplicates
    console.log("[System] Clearing the Pinecone index to start fresh...");
    await pineconeIndex.deleteAll();
    console.log("[System] Index cleared successfully.");

    // Batching documents (Voyage AI Free Tier: 3 RPM / 10K TPM)
    // 30 items @ 35s delay ≈ Safe within 10K TPM and 3 RPM limits
    const BATCH_SIZE = 30;
    
    // Initialize the vector store with the first batch
    const firstBatch = chunkedDocs.slice(0, BATCH_SIZE);
    console.log(`[${new Date().toLocaleTimeString()}] Processing batch 1 of ${Math.ceil(chunkedDocs.length / BATCH_SIZE)}...`);
    const vectorStore = await PineconeStore.fromDocuments(firstBatch, embeddings, {
        pineconeIndex,
    });

    // Process remaining batches with a safe delay (3 RPM = 1 request every 20s, but we use 35s for safety)
    for (let i = BATCH_SIZE; i < chunkedDocs.length; i += BATCH_SIZE) {
        const batch = chunkedDocs.slice(i, i + BATCH_SIZE);
        const currentBatch = Math.floor(i / BATCH_SIZE) + 1;
        const totalBatches = Math.ceil(chunkedDocs.length / BATCH_SIZE);
        
        console.log(`[${new Date().toLocaleTimeString()}] [Rate Limit Wait] Waiting 35 seconds for next batch...`);
        await new Promise(resolve => setTimeout(resolve, 35000)); 
        
        console.log(`[${new Date().toLocaleTimeString()}] Processing batch ${currentBatch} of ${totalBatches}...`);
        await vectorStore.addDocuments(batch);
    }

    console.log("Data Stored successfully")


}

// Only re-index if SKIP_INDEXING is not set to 'true'
if (process.env.SKIP_INDEXING !== 'true') {
    console.log("Starting indexing process...");
    await indexDocument();
} else {
    console.log("✅ Skipping indexing - using existing Pinecone data");
}
