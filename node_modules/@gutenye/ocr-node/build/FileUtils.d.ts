import { FileUtilsBase } from '@gutenye/ocr-common';
export declare class FileUtils extends FileUtilsBase {
    static read(path: string): Promise<string>;
}
