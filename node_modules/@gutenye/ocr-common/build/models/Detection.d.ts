import type { InferenceSession as InferenceSessionCommon } from 'onnxruntime-common';
import type { ModelCreateOptions } from '../types/index.js';
import { ModelBase } from './ModelBase.js';
export declare class Detection extends ModelBase {
    static create({ models, onnxOptions, ...restOptions }: ModelCreateOptions): Promise<Detection>;
    run(path: string, { onnxOptions }?: {
        onnxOptions?: InferenceSessionCommon.RunOptions;
    }): Promise<import("../types/index.js").LineImage[]>;
}
