import { type InferenceSession as InferenceSessionCommon, Tensor } from 'onnxruntime-common';
import type { ImageRaw, LineImage, ModelBaseConstructorArg, ModelBaseOptions, ModelData, ReshapeOptions } from '../types/index.js';
export declare class ModelBase {
    #private;
    options: ModelBaseOptions;
    constructor({ model, options }: ModelBaseConstructorArg);
    runModel({ modelData, onnxOptions, }: {
        modelData: ModelData;
        onnxOptions?: InferenceSessionCommon.RunOptions;
    }): Promise<Tensor>;
    imageToInput(image: ImageRaw, { mean, std }: ReshapeOptions): ModelData;
    debugImage(image: ImageRaw | any, path: string): void;
    debugBoxImage(sourceImage: ImageRaw | any, lineImages: LineImage[], path: string): Promise<void>;
}
