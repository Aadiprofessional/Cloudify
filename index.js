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
    const { psm = '7', scale = 1, invert = false, preprocess = true, autocrop = false, resizeHeight = null, thresholdMax = 200 } = options;

    console.log(`[${requestId}] Running OCR attempt. Options: ${JSON.stringify(options)}`);

    let processedBuffer = imageBuffer;

    if (preprocess) {
        try {
            console.log(`[${requestId}] Preprocessing with Jimp (Scale: ${scale}, Invert: ${invert}, Autocrop: ${autocrop}, ResizeHeight: ${resizeHeight})...`);
            const image = await Jimp.read(imageBuffer);
            
            if (autocrop) {
                image.autocrop();
            }

            if (resizeHeight) {
                image.resize({ h: resizeHeight });
            } else if (scale !== 1) {
                // Resize to specific height, auto width
                // For captchas, height is often the critical factor for Tesseract
                image.resize({ h: 150 }); 
            }

            image.greyscale();

            if (invert) {
                image.invert();
            }

            image.contrast(1).threshold({ max: thresholdMax });
            
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
        const { data } = await worker.recognize(processedBuffer);
        await worker.terminate();

        const baseText = (data.text || '').toUpperCase().replace(/[^A-Z]/g, '').trim();
        let bestText = baseText;
        let bestConf = data.confidence || 0;

        if (Array.isArray(data.words) && data.words.length) {
            const candidates = data.words.map(w => ({
                text: (w.text || '').toUpperCase().replace(/[^A-Z]/g, ''),
                confidence: w.confidence || 0
            })).filter(c => c.text.length >= 3 && c.text.length <= 10);
            candidates.sort((a, b) => (b.confidence || 0) - (a.confidence || 0));
            if (candidates[0] && (candidates[0].confidence || 0) >= bestConf) {
                bestText = candidates[0].text;
                bestConf = candidates[0].confidence || 0;
            }
        }

        if (Array.isArray(data.symbols) && data.symbols.length) {
            const letters = data.symbols.map(s => (s.text || '').toUpperCase()).filter(ch => /^[A-Z]$/.test(ch));
            const avgConf = data.symbols.reduce((sum, s) => sum + (s.confidence || 0), 0) / data.symbols.length;
            const symbolsText = letters.join('');
            if (symbolsText.length >= 3 && symbolsText.length <= 10 && avgConf >= bestConf) {
                bestText = symbolsText;
                bestConf = avgConf;
            }
        }

        console.log(`[${requestId}] OCR Result: '${bestText}' (Confidence: ${bestConf})`);
        return { text: bestText, confidence: bestConf, error: false };
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
            { name: "Standard", options: { psm: '7', scale: 1, preprocess: true } },
            { name: "Resize+Autocrop", options: { psm: '7', scale: 0.5, autocrop: true, preprocess: true } },
            { name: "Raw Buffer", options: { psm: '7', preprocess: false } },
            { name: "Inverted", options: { psm: '7', scale: 1, invert: true, preprocess: true } },
            { name: "Single Word Mode", options: { psm: '8', scale: 1, preprocess: true } },
            { name: "Single Word Resized 100", options: { psm: '8', resizeHeight: 100, autocrop: true, preprocess: true, thresholdMax: 180 } },
            { name: "Single Word Resized 125", options: { psm: '8', resizeHeight: 125, autocrop: true, preprocess: true, thresholdMax: 180 } },
            { name: "Single Word Resized 150", options: { psm: '8', resizeHeight: 150, autocrop: true, preprocess: true, thresholdMax: 180 } }
        ];

        let finalResult = "";

        for (const strategy of strategies) {
            console.log(`[${requestId}] Attempting Strategy: ${strategy.name}`);
            const result = await runOcr(requestId, buffer, strategy.options);
            
            // Validity check:
            // 1. Must have text.
            // 2. Must not be too long (captchas usually < 10 chars). Garbage often is long.
            if (result.text && result.text.length > 0 && result.text.length < 12 && result.confidence >= 10) {
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
