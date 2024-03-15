package org.moussaud.ml;

import java.nio.file.Path;
import java.nio.file.Paths;

import ai.djl.Model;
import ai.djl.basicmodelzoo.cv.classification.ResNetV1;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.Classifications.Classification;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.translator.BaseImageTranslator;
import ai.djl.modality.cv.translator.ImageClassificationTranslator;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.translate.Translator;

public class Doing implements Constants {
    public static void aim(String[] args) throws Exception {
        new Doing().check("data/test-1.png");
        new Doing().check("data/test-2.png");
        new Doing().check("data/test-3.png");
        new Doing().check("data/test-4.png");
    }

    private void check(String path) throws Exception {
        try (Model model = getModel()) {
            Path modelDir = Paths.get("model");
            model.load(modelDir, MUFFIN_VS_CHIHUAHUA_MODEL);

            Translator<Image, Classifications> translator = createTranslator();

            try (Predictor<Image, Classifications> predictor = model.newPredictor(translator)) {
                Path imageFile = Paths.get(path);
                Image img = ImageFactory.getInstance().fromFile(imageFile);
                Classifications classifications = predictor.predict(img);
                System.out.println(classifications.topK());
            }
        }
    }

    private Translator<Image, Classifications> createTranslator() {
        return ImageClassificationTranslator.builder()
                .addTransform(new Resize(224, 224))
                .addTransform(new ToTensor())
                .optApplySoftmax(true).build();
    }

}
