let FileUtils = undefined;
let ImageRaw = undefined;
let InferenceSession = undefined;
let splitIntoLineImages = undefined;
let defaultModels = undefined;
export function registerBackend(backend) {
    FileUtils = backend.FileUtils;
    ImageRaw = backend.ImageRaw;
    InferenceSession = backend.InferenceSession;
    splitIntoLineImages = backend.splitIntoLineImages;
    defaultModels = backend.defaultModels;
}
export { FileUtils, ImageRaw, InferenceSession, splitIntoLineImages, defaultModels };
//# sourceMappingURL=backend.js.map