const fs = require('fs');
const { Jimp } = require('jimp');
const Tesseract = require('tesseract.js');

async function run() {
    try {
        // Read verify.js to get the base64 string
        const verifyContent = fs.readFileSync('verify.js', 'utf8');
        // Match the base64 string. It might be huge, so we need to be careful with regex buffer size,
        // but for a single match it should be fine.
        const match = verifyContent.match(/const validBase64 = "(data:image\/png;base64,[^"]+)";/);
        
        if (!match) {
            console.error('Could not find validBase64 in verify.js');
            return;
        }

        const base64Str = match[1];
        const base64Data = base64Str.replace(/^data:image\/png;base64,/, "");
        const buffer = Buffer.from(base64Data, 'base64');

        // Save original image
        fs.writeFileSync('captcha_original.png', buffer);
        console.log('Saved captcha_original.png');

        // Run Tesseract on original
        // console.log('Running OCR on original...');
        // const resultOriginal = await Tesseract.recognize('captcha_original.png', 'eng');
        // console.log('Original Text:\n', resultOriginal.data.text.trim());
        // console.log('-----------------------------------');

        // Process image with Jimp
        console.log('Processing image...');
        const baseImage = await Jimp.read(buffer);
        console.log(`Original dimensions: ${baseImage.bitmap.width}x${baseImage.bitmap.height}`);
        
        // baseImage.scale(3);
        // baseImage.greyscale();

        const variations = [
            { name: 'Original', fn: img => {} },
            { name: 'Greyscale', fn: img => img.greyscale() },
            { name: 'Red Channel', fn: img => img.color([{ apply: 'red', params: [100] }]).greyscale() }, // Boost red? No, this is not how channel extraction works in Jimp easily.
            // Better way to extract channel in Jimp is to zero out others?
            // Or just use greyscale which averages them.
        ];
        
        // Let's manually manipulate pixels to extract channels?
        // Jimp has `scan`.
        
        // Let's just try basic things first without scaling.
        // Maybe the scaling was the issue.
        
        const manualVariations = [
            { name: 'No Scale', fn: img => {} },
            { name: 'No Scale + Contrast', fn: img => img.contrast(0.5) },
            { name: 'No Scale + Threshold', fn: img => img.greyscale().threshold({ max: 128 }) },
        ];

        for (const v of variations) {
            console.log(`\nRunning Variation: ${v.name}`);
            const img = baseImage.clone();
            v.fn(img);
            
            const buf = await img.getBuffer('image/png');
            // Save for inspection if needed
            // fs.writeFileSync(`captcha_${v.name.replace(/ /g, '_')}.png`, buf);
            
            const res = await Tesseract.recognize(buf, 'eng');
            console.log(`Text: "${res.data.text.trim().replace(/\n/g, ' ')}"`);
            console.log(`Confidence: ${res.data.confidence}`);
        }

    } catch (error) {
        console.error('Error:', error);
    }
}

run();
