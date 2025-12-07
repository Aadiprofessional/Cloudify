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
    const { psm = '7', scale = 1, invert = false, preprocess = true, autocrop = false } = options;

    console.log(`[${requestId}] Running OCR attempt. Options: ${JSON.stringify(options)}`);

    let processedBuffer = imageBuffer;

    if (preprocess) {
        try {
            console.log(`[${requestId}] Preprocessing with Jimp (Scale: ${scale}, Invert: ${invert}, Autocrop: ${autocrop})...`);
            const image = await Jimp.read(imageBuffer);
            
            if (autocrop) {
                image.autocrop();
            }

            if (scale !== 1) {
                // Resize to specific height, auto width
                // For captchas, height is often the critical factor for Tesseract
                image.resize({ h: 150 }); 
            }

            image.greyscale();

            if (invert) {
                image.invert();
            }

            // A bit of blur can help reduce single-pixel noise
            // image.blur(1); 
            image.contrast(1).threshold({ max: 200 });
            
            processedBuffer = await image.getBuffer('image/png');
        } catch (error) {
            console.error(`[${requestId}] Jimp preprocessing failed:`, error.message);
            // If Jimp fails, we return null to signal failure, or we could fallback to raw buffer.
            // But let's let the loop handle fallbacks.
            // For now, if preprocessing was requested but failed, we might want to try raw buffer in this same attempt?
            // No, better to fail this attempt and let the retry logic pick the "raw" strategy.
            return { text: '', confidence: 0, error: true };
        }
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

    try {
        const { data: { text, confidence } } = await worker.recognize(processedBuffer);
        await worker.terminate();

        const result = text.trim().replace(/[^A-Z]/g, ''); // Ensure only uppercase letters
        console.log(`[${requestId}] OCR Result: '${result}' (Confidence: ${confidence})`);
        
        return { text: result, confidence, error: false };
    } catch (ocrError) {
        console.error(`[${requestId}] Tesseract execution failed:`, ocrError.message);
        await worker.terminate();
        return { text: '', confidence: 0, error: true };
    }
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

        // Define strategies
        const strategies = [
            // 1. Standard: High contrast, PSM 7 (Single Line)
            { name: "Standard", options: { psm: '7', scale: 1, preprocess: true } },
            
            // 2. Resized + Autocrop: Good for weirdly sized images. Height 150 is usually sweet spot for OCR.
            { name: "Resize+Autocrop", options: { psm: '7', scale: 0.5, autocrop: true, preprocess: true } }, // scale!=1 triggers resize logic
            
            // 3. Raw Buffer: Bypass Jimp entirely if it's crashing or messing up. PSM 7.
            { name: "Raw Buffer", options: { psm: '7', preprocess: false } },

            // 4. Inverted: Handle dark background/light text.
            { name: "Inverted", options: { psm: '7', scale: 1, invert: true, preprocess: true } },

            // 5. PSM 8 (Single Word): Force Tesseract to find a single word. Good if there's noise.
            { name: "Single Word Mode", options: { psm: '8', scale: 1, preprocess: true } }
        ];

        let finalResult = "";

        for (const strategy of strategies) {
            console.log(`[${requestId}] Attempting Strategy: ${strategy.name}`);
            const result = await runOcr(requestId, buffer, strategy.options);
            
            // Validity check:
            // 1. Must have text.
            // 2. Must not be too long (captchas usually < 10 chars). Garbage often is long.
            if (result.text && result.text.length > 0 && result.text.length < 12) {
                finalResult = result.text;
                console.log(`[${requestId}] Success with strategy: ${strategy.name}`);
                break;
            } else {
                console.log(`[${requestId}] Strategy ${strategy.name} failed or produced invalid result ('${result.text}').`);
            }
        }

        console.log(`[${requestId}] Final Extracted text: '${finalResult}'`);
        res.json({ solution: finalResult });
        console.log(`[${requestId}] Response sent: { solution: '${finalResult}' }`);

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
