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
            const raw = data.symbols.map(s => ({
                ch: (s.text || '').toUpperCase(),
                conf: s.confidence || 0,
                bbox: s.bbox || s
            })).filter(s => /^[A-Z]$/.test(s.ch));

            const heights = raw.map(s => {
                if (s.bbox && typeof s.bbox.h === 'number') return s.bbox.h;
                if (s.bbox && typeof s.bbox.y1 === 'number' && typeof s.bbox.y0 === 'number') return Math.abs(s.bbox.y1 - s.bbox.y0);
                return 0;
            }).filter(h => h > 0);

            const sortedH = heights.slice().sort((a, b) => a - b);
            const medianH = sortedH.length ? sortedH[Math.floor(sortedH.length / 2)] : 0;
            const minConf = 10;

            const positioned = raw.map(s => ({
                ch: s.ch,
                conf: s.conf,
                h: (s.bbox && typeof s.bbox.h === 'number') ? s.bbox.h : medianH,
                x: (s.bbox && typeof s.bbox.x === 'number') ? s.bbox.x : (s.bbox && typeof s.bbox.x0 === 'number' ? s.bbox.x0 : 0)
            }));

            const selected = positioned
                .filter(s => s.conf >= minConf && (medianH ? s.h >= medianH * 0.6 : true))
                .sort((a, b) => a.x - b.x);

            let candidate = selected.map(s => s.ch).join('');
            const avgConfSel = selected.length ? selected.reduce((sum, s) => sum + s.conf, 0) / selected.length : 0;

            if (selected.length >= 2 && selected[0].ch === 'C' && selected[0].conf < 12) {
                candidate = selected.slice(1).map(s => s.ch).join('');
            }

            const lenBonus = candidate.length === 4 ? 10 : candidate.length === 5 ? 6 : candidate.length === 6 ? 4 : candidate.length === 3 ? 2 : 0;
            const candidateScore = avgConfSel + lenBonus;
            const bestScore = (bestConf || 0) + (bestText.length === 4 ? 10 : 0);

            if (candidate.length >= 3 && candidate.length <= 10 && candidateScore >= bestScore) {
                bestText = candidate;
                bestConf = avgConfSel;
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

const runOcrSegmented = async (requestId, imageBuffer, options = {}) => {
    const { invert = false, thresholdMax = 180, resizeHeight = 150, autocrop = true } = options;
    let processed;
    try {
        const base = await Jimp.read(imageBuffer);
        if (autocrop) base.autocrop();
        if (resizeHeight) base.resize({ h: resizeHeight });
        base.greyscale();
        if (invert) base.invert();
        base.contrast(1).threshold({ max: thresholdMax });
        processed = base;
    } catch (e) {
        return { text: '', confidence: 0, error: true };
    }

    const w = processed.bitmap.width;
    const h = processed.bitmap.height;
    const buf = processed.bitmap.data;
    const proj = new Array(w).fill(0);
    for (let x = 0; x < w; x++) {
        let s = 0;
        for (let y = 0; y < h; y++) {
            const i = (w * y + x) * 4;
            const v = buf[i];
            if (v < 128) s++;
        }
        proj[x] = s;
    }
    const th = Math.max(1, Math.floor(h * 0.02));
    const segs = [];
    let start = -1;
    for (let x = 0; x < w; x++) {
        if (proj[x] > th) {
            if (start === -1) start = x;
        } else if (start !== -1) {
            const end = x - 1;
            const width = end - start + 1;
            if (width >= Math.max(6, Math.floor(w * 0.02))) segs.push({ start, end, width });
            start = -1;
        }
    }
    if (start !== -1) {
        const end = w - 1;
        const width = end - start + 1;
        if (width >= Math.max(6, Math.floor(w * 0.02))) segs.push({ start, end, width });
    }
    if (segs.length > 6) {
        const merged = [];
        let curr = segs[0];
        for (let i = 1; i < segs.length; i++) {
            const gap = segs[i].start - curr.end;
            if (gap <= 3) {
                curr.end = segs[i].end;
                curr.width = curr.end - curr.start + 1;
            } else {
                merged.push(curr);
                curr = segs[i];
            }
        }
        merged.push(curr);
        segs.splice(0, segs.length, ...merged);
    }
    segs.sort((a, b) => a.start - b.start);
    const take = Math.min(5, Math.max(4, segs.length));
    const chosen = segs.slice(0, take);

    const worker = await createWorker('eng', 1, {
        cachePath: path.join('/tmp', 'eng.traineddata.gz'),
        cacheMethod: 'refresh'
    });
    await worker.setParameters({
        tessedit_char_whitelist: 'ABCDEFGHIJKLMNOPQRSTUVWXYZ',
        tessedit_pageseg_mode: '10'
    });

    const letters = [];
    for (const seg of chosen) {
        const crop = processed.clone().crop(seg.start, 0, seg.end - seg.start + 1, h);
        crop.autocrop();
        const b = await crop.getBuffer('image/png');
        const { data } = await worker.recognize(b);
        const ch = (data.text || '').toUpperCase().replace(/[^A-Z]/g, '').trim();
        const conf = data.confidence || 0;
        if (!ch) continue;
        letters.push({ ch: ch[0], conf, width: seg.width });
    }
    await worker.terminate();

    if (!letters.length) return { text: '', confidence: 0, error: false };
    const widths = letters.map(l => l.width).slice().sort((a, b) => a - b);
    const medW = widths.length ? widths[Math.floor(widths.length / 2)] : 0;
    let result = letters.slice();
    if (result.length >= 5 && result[0].ch === 'C' && result[0].conf < 12) {
        result = result.slice(1);
    }
    while (result.length > 4) {
        let idx = 0;
        for (let i = 1; i < result.length; i++) {
            if (result[i].conf < result[idx].conf) idx = i;
        }
        result.splice(idx, 1);
    }
    result = result.filter(l => l.conf >= 8 && (medW ? l.width >= medW * 0.6 : true));
    const text = result.map(l => l.ch).join('');
    const confAvg = result.length ? Math.round(result.reduce((s, l) => s + l.conf, 0) / result.length) : 0;
    return { text, confidence: confAvg, error: false };
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
            { name: "Single Word Resized 150", options: { psm: '8', resizeHeight: 150, autocrop: true, preprocess: true, thresholdMax: 180 } },
            { name: "Segmented", options: { preprocess: true, autocrop: true, resizeHeight: 150, thresholdMax: 180 } }
        ];

        let finalResult = "";

        for (const strategy of strategies) {
            console.log(`[${requestId}] Attempting Strategy: ${strategy.name}`);
            const result = strategy.name === "Segmented"
                ? await runOcrSegmented(requestId, buffer, strategy.options)
                : await runOcr(requestId, buffer, strategy.options);
            
            // Validity check:
            // 1. Must have text.
            // 2. Must not be too long (captchas usually < 10 chars). Garbage often is long.
            const confOk = result.confidence >= 10 || (result.text && result.text.length === 4);
            if (result.text && result.text.length > 0 && result.text.length < 12 && confOk) {
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
