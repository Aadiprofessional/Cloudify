const express = require('express');
const bodyParser = require('body-parser');
const { createWorker } = require('tesseract.js');
const { Jimp } = require('jimp');

const path = require('path');
const fs = require('fs');

const app = express();
const port = 3000;

// Middleware to parse JSON bodies. Increase limit for base64 images.
app.use(bodyParser.json({ limit: '10mb' }));

// Handler function for OCR
const handleOcr = async (req, res) => {
    const requestId = Date.now().toString();
    console.log(`[${requestId}] Request received`);
    console.log(`[${requestId}] Method: ${req.method}, Path: ${req.path}`);
    
    try {
        if (!req.body) {
             console.log(`[${requestId}] No body provided`);
             return res.json({ solution: "" });
        }
        
        // Log body keys (avoid logging full base64 string to keep logs clean)
        console.log(`[${requestId}] Body keys: ${Object.keys(req.body).join(', ')}`);

        // Always ensure we return JSON with solution key, even for empty/invalid inputs
        if (!req.body.captcha) {
            console.log(`[${requestId}] 'captcha' field missing in body`);
            return res.json({ solution: "" });
        }

        const { captcha } = req.body;
        console.log(`[${requestId}] Captcha length: ${captcha.length}`);

        console.log(`[${requestId}] Processing captcha...`);

        // Remove header if present to get pure base64
        const base64Data = captcha.replace(/^data:image\/\w+;base64,/, "");
        const buffer = Buffer.from(base64Data, 'base64');
        console.log(`[${requestId}] Base64 decoded, buffer length: ${buffer.length}`);

        // Preprocess with Jimp
        console.log(`[${requestId}] Starting Jimp processing...`);
        const image = await Jimp.read(buffer);
        console.log(`[${requestId}] Image dimensions: ${image.bitmap.width}x${image.bitmap.height}`);
        
        image.greyscale().contrast(1).threshold({ max: 200 });
        const processedBuffer = await image.getBuffer('image/png');
        console.log(`[${requestId}] Jimp processing complete. Processed buffer length: ${processedBuffer.length}`);

        // OCR with Tesseract
        console.log(`[${requestId}] Starting Tesseract...`);
        // Use /tmp for cache on serverless environments
        const worker = await createWorker('eng', 1, {
            cachePath: path.join('/tmp', 'eng.traineddata.gz'),
            cacheMethod: 'refresh', // Force refresh or use 'readOnly' if we ship the file
            logger: m => console.log(`[${requestId}] [Tesseract] ${m.status}: ${m.progress}`)
        });
        
        console.log(`[${requestId}] Tesseract worker created. Setting parameters...`);
        await worker.setParameters({
            tessedit_char_whitelist: 'ABCDEFGHIJKLMNOPQRSTUVWXYZ',
            tessedit_pageseg_mode: '7',
        });

        console.log(`[${requestId}] Recognizing text...`);
        const { data: { text, confidence } } = await worker.recognize(processedBuffer);
        console.log(`[${requestId}] Recognition complete. Confidence: ${confidence}`);
        await worker.terminate();
        console.log(`[${requestId}] Tesseract worker terminated`);

        const extractedText = text.trim();
        console.log(`[${requestId}] Extracted text: '${extractedText}'`);

        res.json({ solution: extractedText });
        console.log(`[${requestId}] Response sent: { solution: '${extractedText}' }`);

    } catch (error) {
        console.error(`[${requestId}] Error processing captcha:`, error);
        // Even on error, return empty solution to satisfy strict requirements
        res.json({ solution: "" });
        console.log(`[${requestId}] Error response sent: { solution: "" }`);
    }
};

app.post('/', handleOcr);
app.post('/solve', handleOcr);

if (require.main === module) {
    app.listen(port, () => {
        console.log(`OCR Service listening at http://localhost:${port}`);
    });
}

module.exports = app;
