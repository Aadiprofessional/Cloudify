const http = require('http');
const fs = require('fs');
const app = require('./index.js');

const PORT = 3001;
const server = app.listen(PORT, () => {
    console.log(`Test server running on port ${PORT}`);
    
    // Extract base64 from verify.js
    const verifyContent = fs.readFileSync('verify.js', 'utf8');
    const match = verifyContent.match(/const validBase64 = "(.*)"/);
    
    if (!match) {
        console.error('Could not find base64 string in verify.js');
        server.close();
        process.exit(1);
    }
    
    const validBase64 = match[1];
    const postData = JSON.stringify({
        captcha: validBase64
    });

    const options = {
        hostname: 'localhost',
        port: PORT,
        path: '/',
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Content-Length': Buffer.byteLength(postData)
        }
    };

    const req = http.request(options, (res) => {
        console.log(`StatusCode: ${res.statusCode}`);
        let body = '';
        res.on('data', chunk => body += chunk);
        res.on('end', () => {
            console.log('Response:', body);
            server.close();
        });
    });

    req.on('error', (e) => {
        console.error(`Problem with request: ${e.message}`);
        server.close();
    });

    req.write(postData);
    req.end();
});
