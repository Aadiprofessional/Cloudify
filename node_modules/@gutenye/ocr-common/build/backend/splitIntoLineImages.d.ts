import type { LineImage, ImageRaw as ImageRawType } from '../types/index.js';
export declare function splitIntoLineImages(image: ImageRawType, sourceImage: ImageRawType): Promise<LineImage[]>;
