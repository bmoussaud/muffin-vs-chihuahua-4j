package org.moussaud.ml;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.translator.ImageClassificationTranslator;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import jakarta.annotation.PostConstruct;
import jakarta.annotation.PreDestroy;

@Component
public class MachineLearningService implements Constants {

    private Model model;
    private Translator<Image, Classifications> translator;

    static Logger logger = LoggerFactory.getLogger(MachineLearningService.class);

    @PostConstruct
    public void init() {
        logger.info("Initialize the model {}", MUFFIN_VS_CHIHUAHUA_MODEL);
        model = getModel();
        Path modelDir = Paths.get("model");
        try {
            model.load(modelDir, MUFFIN_VS_CHIHUAHUA_MODEL);
        } catch (Exception e) {
            var msg = String.format("Cannot load model %s from %s ", MUFFIN_VS_CHIHUAHUA_MODEL, modelDir);
            logger.error(msg, e);
            throw new RuntimeException(msg, e);
        }

        translator = ImageClassificationTranslator.builder()
                .addTransform(new Resize(224, 224))
                .addTransform(new ToTensor())
                .optApplySoftmax(true).build();

    }

    @PreDestroy
    public void closeModel() {
        logger.info("Close the model {}", MUFFIN_VS_CHIHUAHUA_MODEL);
        model.close();
    }

    public Classifications check(String path) {
        try (Predictor<Image, Classifications> predictor = model.newPredictor(translator)) {
            try {
                Path imageFile = Paths.get(path);
                Image img = ImageFactory.getInstance().fromFile(imageFile);
                Classifications classifications;
                classifications = predictor.predict(img);
                return classifications;
            } catch (Exception e) {
                var msg = String.format("Cannot check image %s", path);
                logger.error(msg, e);
                throw new RuntimeException(msg, e);
            }

        }
    }

}
