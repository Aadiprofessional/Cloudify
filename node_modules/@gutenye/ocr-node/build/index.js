import fs from 'node:fs/promises';
import BaseOcr, { registerBackend } from '@gutenye/ocr-common';
import { splitIntoLineImages } from '@gutenye/ocr-common/splitIntoLineImages';
import defaultModels from '@gutenye/ocr-models/node';
import { InferenceSession } from 'onnxruntime-node';
import { FileUtils } from './FileUtils.js';
import { ImageRaw } from './ImageRaw.js';
registerBackend({
    FileUtils,
    ImageRaw,
    InferenceSession,
    splitIntoLineImages,
    defaultModels,
});
// biome-ignore lint/complexity/noStaticOnlyClass: <explanation>
class Ocr extends BaseOcr {
    static async create(options = {}) {
        const ocr = await BaseOcr.create(options);
        if (options.debugOutputDir) {
            await fs.mkdir(options.debugOutputDir, { recursive: true });
        }
        return ocr;
    }
}
export * from '@gutenye/ocr-common';
export default Ocr;
//# sourceMappingURL=index.js.map