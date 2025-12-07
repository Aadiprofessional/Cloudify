import type { ImageRawData } from '../types/index.js';
export declare class ImageRawBase {
    data: ImageRawData['data'];
    width: ImageRawData['width'];
    height: ImageRawData['height'];
    constructor({ data, width, height }: ImageRawData);
    getImageRawData(): ImageRawData;
}
