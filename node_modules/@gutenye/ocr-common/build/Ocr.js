import { Detection, Recognition } from './models/index.js';
export class Ocr {
    static async create(options = {}) {
        const detection = await Detection.create(options);
        const recognition = await Recognition.create(options);
        return new Ocr({ detection, recognition });
    }
    #detection;
    #recognition;
    constructor({ detection, recognition, }) {
        this.#detection = detection;
        this.#recognition = recognition;
    }
    async detect(image, options = {}) {
        const lineImages = await this.#detection.run(image, options);
        const texts = await this.#recognition.run(lineImages, options);
        return texts;
    }
}
//# sourceMappingURL=Ocr.js.map