import fs from 'node:fs/promises';
import { FileUtilsBase } from '@gutenye/ocr-common';
export class FileUtils extends FileUtilsBase {
    static async read(path) {
        return await fs.readFile(path, 'utf8');
    }
}
//# sourceMappingURL=FileUtils.js.map