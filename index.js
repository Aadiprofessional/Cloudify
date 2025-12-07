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

const runOcr = async (requestId, imageBuffer, options = {}) => {
    const { psm = '7', scale = 1, invert = false, preprocess = true } = options;

    console.log(`[${requestId}] Running OCR attempt. Options: ${JSON.stringify(options)}`);

    let processedBuffer = imageBuffer;

    if (preprocess) {
        console.log(`[${requestId}] Preprocessing with Jimp (Scale: ${scale}, Invert: ${invert})...`);
        const image = await Jimp.read(imageBuffer);
        
        if (scale !== 1) {
            image.resize({ h: 100 }); // Resize to height 100, auto width
        }

        image.greyscale();

        if (invert) {
            image.invert();
        }

        image.contrast(1).threshold({ max: 200 });
        processedBuffer = await image.getBuffer('image/png');
    }

    // Use /tmp for cache on serverless environments
    const worker = await createWorker('eng', 1, {
        cachePath: path.join('/tmp', 'eng.traineddata.gz'),
        cacheMethod: 'refresh',
        logger: m => {
             // Reduce log spam, only log major status changes
             if (m.status === 'recognizing text' && m.progress % 0.5 === 0) {
                 console.log(`[${requestId}] [Tesseract] ${m.status}: ${m.progress}`);
             }
        }
    });

    await worker.setParameters({
        tessedit_char_whitelist: 'ABCDEFGHIJKLMNOPQRSTUVWXYZ',
        tessedit_pageseg_mode: psm,
    });

    const { data: { text, confidence } } = await worker.recognize(processedBuffer);
    await worker.terminate();

    const result = text.trim().replace(/[^A-Z]/g, ''); // Ensure only uppercase letters
    console.log(`[${requestId}] OCR Result: '${result}' (Confidence: ${confidence})`);
    
    return { text: result, confidence };
};

// Handler function for OCR
const handleOcr = async (req, res) => {
    const requestId = Date.now().toString();
    console.log(`[${requestId}] Request received at ${new Date().toISOString()}`);
    console.log(`[${requestId}] Method: ${req.method}, Path: ${req.path}`);
    
    try {
        if (!req.body || !req.body.captcha) {
             console.log(`[${requestId}] No body or captcha provided`);
             return res.json({ solution: "" });
        }

        const { captcha } = req.body;
        console.log(`[${requestId}] Captcha length: ${captcha.length}`);
        
        // --- LOGGING FULL BASE64 AS REQUESTED ---
        console.log(`[${requestId}] FULL CAPTCHA BASE64: ${captcha}`);
        // ----------------------------------------

        // Remove header if present to get pure base64
        const base64Data = captcha.replace(/^data:image\/\w+;base64,/, "");
        const buffer = Buffer.from(base64Data, 'base64');
        console.log(`[${requestId}] Base64 decoded, buffer length: ${buffer.length}`);

        // --- ATTEMPT 1: Original Settings (Optimized for "ATLK" style) ---
        // PSM 7 (Single Line), No Scaling (unless huge?), High Contrast
        let result = await runOcr(requestId, buffer, { psm: '7', scale: 1, preprocess: true });

        // --- ATTEMPT 2: Resize if empty ---
        // If result is empty, maybe image is too big/small.
        // The failing image was 910x324. Resizing to height 100 usually helps Tesseract.
        if (!result.text) {
             console.log(`[${requestId}] Attempt 1 failed. Retrying with resizing...`);
             result = await runOcr(requestId, buffer, { psm: '7', scale: 0.5, preprocess: true }); // Scale triggers resize logic
        }

        // --- ATTEMPT 3: Invert Colors ---
        if (!result.text) {
             console.log(`[${requestId}] Attempt 2 failed. Retrying with inverted colors...`);
             result = await runOcr(requestId, buffer, { psm: '7', scale: 1, invert: true, preprocess: true });
        }
        
        // --- ATTEMPT 4: PSM 6 (Assume block of text) ---
        if (!result.text) {
            console.log(`[${requestId}] Attempt 3 failed. Retrying with PSM 6...`);
            result = await runOcr(requestId, buffer, { psm: '6', scale: 1, preprocess: true });
        }

        console.log(`[${requestId}] Final Extracted text: '${result.text}'`);
        res.json({ solution: result.text });
        console.log(`[${requestId}] Response sent: { solution: '${result.text}' }`);

    } catch (error) {
        console.error(`[${requestId}] Error processing captcha:`, error);
        res.json({ solution: "" });
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
