import type { ModelCreateOptions } from './types/index.js';
import { Detection, Recognition } from './models/index.js';
export declare class Ocr {
    #private;
    static create(options?: ModelCreateOptions): Promise<Ocr>;
    constructor({ detection, recognition, }: {
        detection: Detection;
        recognition: Recognition;
    });
    detect(image: string, options?: {}): Promise<import("./types/index.js").Line[]>;
}
