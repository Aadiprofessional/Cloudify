import type { InferenceSession as InferenceSessionCommon, Tensor } from 'onnxruntime-common';
import type { Dictionary, Line, LineImage, ModelBaseConstructorArg, ModelCreateOptions } from '../types/index.js';
import { ModelBase } from './ModelBase.js';
export declare class Recognition extends ModelBase {
    #private;
    static create({ models, onnxOptions, ...restOptions }: ModelCreateOptions): Promise<Recognition>;
    constructor(options: ModelBaseConstructorArg, dictionary: Dictionary);
    run(lineImages: LineImage[], { onnxOptions }?: {
        onnxOptions?: InferenceSessionCommon.RunOptions;
    }): Promise<Line[]>;
    decodeText(output: Tensor): Line[];
}
