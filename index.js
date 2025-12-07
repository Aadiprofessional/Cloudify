const express = require('express');
const bodyParser = require('body-parser');
const { createWorker } = require('tesseract.js');
const { Jimp } = require('jimp');

const app = express();
const port = 3000;

// Middleware to parse JSON bodies. Increase limit for base64 images.
app.use(bodyParser.json({ limit: '10mb' }));

app.post('/', async (req, res) => {
    try {
        const { captcha } = req.body;

        if (!captcha) {
            return res.status(400).json({ error: 'Missing captcha field' });
        }

        console.log('Processing captcha...');

        // Remove header if present to get pure base64
        const base64Data = captcha.replace(/^data:image\/\w+;base64,/, "");
        const buffer = Buffer.from(base64Data, 'base64');

        // Preprocess with Jimp
        const image = await Jimp.read(buffer);
        image.greyscale().contrast(1).threshold({ max: 200 });
        const processedBuffer = await image.getBuffer('image/png');

        // OCR with Tesseract
        const worker = await createWorker('eng');
        await worker.setParameters({
            tessedit_char_whitelist: 'ABCDEFGHIJKLMNOPQRSTUVWXYZ',
            tessedit_pageseg_mode: '7',
        });

        const { data: { text } } = await worker.recognize(processedBuffer);
        await worker.terminate();

        const extractedText = text.trim();
        console.log('Extracted text:', extractedText);

        res.json({ solution: extractedText });

    } catch (error) {
        console.error('Error processing captcha:', error);
        res.status(500).json({ error: 'Failed to process captcha' });
    }
});

app.listen(port, () => {
    console.log(`OCR Service listening at http://localhost:${port}`);
});
