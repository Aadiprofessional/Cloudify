import type { FileUtils as FileUtilsType, ImageRaw as ImageRawType, InferenceSession as InferenceSessionType, ModelCreateOptions as ModelCreateOptionsType, SplitIntoLineImages as SplitIntoLineImagesType } from '../types/index.js';
declare let FileUtils: FileUtilsType | any;
declare let ImageRaw: ImageRawType | any;
declare let InferenceSession: InferenceSessionType | any;
declare let splitIntoLineImages: SplitIntoLineImagesType;
declare let defaultModels: ModelCreateOptionsType['models'];
export declare function registerBackend(backend: {
    FileUtils: FileUtilsType;
    ImageRaw: ImageRawType | any;
    InferenceSession: InferenceSessionType | any;
    splitIntoLineImages: SplitIntoLineImagesType;
    defaultModels: ModelCreateOptionsType['models'];
}): void;
export { FileUtils, ImageRaw, InferenceSession, splitIntoLineImages, defaultModels };
