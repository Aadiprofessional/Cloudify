import { ImageRawBase } from '@gutenye/ocr-common';
import type { ImageRawData, LineImage, SizeOption } from '@gutenye/ocr-common';
import sharp from 'sharp';
export declare class ImageRaw extends ImageRawBase {
    #private;
    static open(path: string): Promise<ImageRaw>;
    constructor(imageRawData: ImageRawData);
    write(path: string): Promise<sharp.OutputInfo>;
    resize(size: SizeOption): Promise<this>;
    drawBox(lineImages: LineImage[]): Promise<this>;
}
