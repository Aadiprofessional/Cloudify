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
    const { psm = '7', scale = 1, invert = false, preprocess = true, threshold = true } = options;

    console.log(`[${requestId}] Running OCR attempt. Options: ${JSON.stringify(options)}`);

    let processedBuffer = imageBuffer;

    if (preprocess) {
        try {
            console.log(`[${requestId}] Preprocessing with Jimp (Scale: ${scale}, Invert: ${invert}, Threshold: ${threshold})...`);
            // Clone buffer to avoid stream locks or mutation issues
            const bufferCopy = Buffer.from(imageBuffer);
            const image = await Jimp.read(bufferCopy);
            
            if (scale !== 1) {
                // If scale is 0.5, we resize to half height. If > 1, double.
                // Assuming standard captcha height ~50-100px.
                // Let's rely on fixed height resizing for consistency if scale is passed as a "target height" flag, 
                // but here let's stick to multipliers if provided, OR fixed height strategies.
                // Strategy: If scale < 1, it's a downsample. If > 1, upsample.
                
                const currentW = image.bitmap.width;
                const currentH = image.bitmap.height;
                
                let newW = Math.floor(currentW * scale);
                let newH = Math.floor(currentH * scale);
                
                // Safety check
                if (newW < 1) newW = 1;
                if (newH < 1) newH = 1;

                image.resize({ w: newW, h: newH }); 
            }

            image.greyscale();

            if (invert) {
                image.invert();
            }

            image.contrast(1); // Increase contrast
            
            if (threshold) {
                image.threshold({ max: 200 });
            }

            processedBuffer = await image.getBuffer('image/png');
        } catch (err) {
            console.error(`[${requestId}] Jimp preprocessing failed:`, err.message);
            // Fallback to original buffer if preprocessing fails
            processedBuffer = imageBuffer; 
        }
    }

    // Use /tmp for cache on serverless environments
    const worker = await createWorker('eng', 1, {
        cachePath: path.join('/tmp', 'eng.traineddata.gz'),
        cacheMethod: 'refresh',
        logger: m => {
             // Reduce log spam
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

        let bestResult = { text: "", confidence: 0 };

        // Helper to check if result is good enough
        const isAcceptable = (res) => {
            // Confidence threshold: 50. Text length: 3-6 chars (standard captcha)
            return res.text.length >= 3 && res.text.length <= 8 && res.confidence > 50;
        };

        // --- ATTEMPT 1: Standard (Optimized for "ATLK" style) ---
        // PSM 7 (Single Line), No Scaling, High Contrast + Threshold
        let result = await runOcr(requestId, buffer, { psm: '7', scale: 1, preprocess: true, threshold: true });
        if (isAcceptable(result)) {
             res.json({ solution: result.text });
             console.log(`[${requestId}] Success on Attempt 1. Response sent: { solution: '${result.text}' }`);
             return;
        }
        if (result.confidence > bestResult.confidence) bestResult = result;

        // --- ATTEMPT 2: Resize (Downscale for large images) ---
        // The failing image was large. Let's try 0.5 scale.
        console.log(`[${requestId}] Attempt 1 weak. Retrying with resizing (0.5x)...`);
        result = await runOcr(requestId, buffer, { psm: '7', scale: 0.5, preprocess: true, threshold: true });
        if (isAcceptable(result)) {
             res.json({ solution: result.text });
             console.log(`[${requestId}] Success on Attempt 2. Response sent: { solution: '${result.text}' }`);
             return;
        }
        if (result.confidence > bestResult.confidence) bestResult = result;

        // --- ATTEMPT 3: Soft Processing (No Threshold) ---
        // Sometimes thresholding kills faint text. Just contrast.
        console.log(`[${requestId}] Attempt 2 weak. Retrying without thresholding...`);
        result = await runOcr(requestId, buffer, { psm: '7', scale: 1, preprocess: true, threshold: false });
        if (isAcceptable(result)) {
             res.json({ solution: result.text });
             console.log(`[${requestId}] Success on Attempt 3. Response sent: { solution: '${result.text}' }`);
             return;
        }
        if (result.confidence > bestResult.confidence) bestResult = result;

        // --- ATTEMPT 4: Invert Colors ---
        console.log(`[${requestId}] Attempt 3 weak. Retrying with inverted colors...`);
        result = await runOcr(requestId, buffer, { psm: '7', scale: 1, invert: true, preprocess: true, threshold: true });
        if (isAcceptable(result)) {
             res.json({ solution: result.text });
             console.log(`[${requestId}] Success on Attempt 4. Response sent: { solution: '${result.text}' }`);
             return;
        }
        if (result.confidence > bestResult.confidence) bestResult = result;
        
        // --- ATTEMPT 5: PSM 6 (Block) as last resort ---
        // Only if we have nothing decent yet.
        console.log(`[${requestId}] Attempt 4 weak. Retrying with PSM 6...`);
        result = await runOcr(requestId, buffer, { psm: '6', scale: 1, preprocess: true, threshold: true });
        if (isAcceptable(result)) {
             res.json({ solution: result.text });
             console.log(`[${requestId}] Success on Attempt 5. Response sent: { solution: '${result.text}' }`);
             return;
        }
        if (result.confidence > bestResult.confidence) bestResult = result;

        // If we reached here, none were "Acceptable" by strict standards.
        // Return the best one we found, or empty if confidence is garbage (<20).
        const finalSolution = bestResult.confidence > 30 ? bestResult.text : "";
        console.log(`[${requestId}] All attempts finished. Best Result: '${bestResult.text}' (Conf: ${bestResult.confidence}). Returning: '${finalSolution}'`);
        
        res.json({ solution: finalSolution });
        console.log(`[${requestId}] Final Response sent: { solution: '${finalSolution}' }`);

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
