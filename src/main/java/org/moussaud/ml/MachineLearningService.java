package org.moussaud.ml;

import java.io.IOException;
import java.io.Writer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import ai.djl.Model;
import ai.djl.basicdataset.cv.classification.ImageFolder;
import ai.djl.basicmodelzoo.cv.classification.ResNetV1;
import ai.djl.inference.Predictor;
import ai.djl.metric.Metrics;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.translator.ImageClassificationTranslator;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.Translator;
import jakarta.annotation.PostConstruct;
import jakarta.annotation.PreDestroy;

@Component
public class MachineLearningService {

    static Logger logger = LoggerFactory.getLogger(MachineLearningService.class);

    private Model model;
    private Translator<Image, Classifications> translator;

    private Path modelDir = Paths.get("model");
    private String modelName = Constants.MUFFIN_VS_CHIHUAHUA_MODEL;

    @PostConstruct
    public void init() {

        model = getModel();

        translator = ImageClassificationTranslator.builder()
                .addTransform(new Resize(224, 224))
                .addTransform(new ToTensor())
                .optApplySoftmax(true).build();

    }

    @PreDestroy
    public void closeModel() {
        logger.info("Close the model {}", model.getName());
        model.close();
    }

    private Model getModel() {
        logger.info("get the model {}", modelName);
        Model model = Model.newInstance(modelName);
        Block resNetN50 = ResNetV1.builder()
                .setImageShape(new Shape(3, 224, 224))
                .setNumLayers(50)
                .setOutSize(2)
                .build();

        model.setBlock(resNetN50);
        return model;
    }

    private void loadModel() {
        logger.info("load the model {}", modelName);
        try {
            model.load(modelDir, modelName);
        } catch (Exception e) {
            var msg = String.format("Cannot load model %s from %s ", modelName, modelDir);
            logger.error(msg, e);
            throw new RuntimeException(msg, e);
        }
    }

    public Classifications check(String path) {
        logger.info("check {}", path);
        this.loadModel();
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

    public void learn(String pathToTrain, String pathToValidate) {
        logger.info("learn {} - {}", pathToTrain, pathToValidate);
        try {
            var dataset = loadDataSet(pathToTrain);
            var validationset = loadDataSet(pathToValidate);

            var trainingConfing = new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss())
                    .addEvaluator(new Accuracy())
                    .optExecutorService()
                    .addTrainingListeners(TrainingListener.Defaults.logging(1));

            try (Model model = getModel();
                    Trainer trainer = model.newTrainer(trainingConfing)) {
                trainer.setMetrics(new Metrics());
                trainer.initialize(new Shape(1, 3, 244, 244));
                logger.info("Fit the model " + model.getName());
                EasyTrain.fit(trainer, 1, dataset, validationset);

                logger.info("Persist the model " + model.getName());
                model.save(modelDir, model.getName());
                saveLabels(dataset.getSynset());
            }
        } catch (Exception e) {
            var msg = String.format("learn failed %s,%s", pathToTrain, pathToValidate);
            logger.error(msg, e);
            throw new RuntimeException(msg, e);
        }
    }

    private void saveLabels(List<String> synset) throws IOException {
        var labelFile = modelDir.resolve("synset.txt");
        try (Writer writer = Files.newBufferedWriter(labelFile)) {
            writer.write(String.join("\n", synset));
        }
    }

    private ImageFolder loadDataSet(String folder) throws IOException {
        ImageFolder dataset = ImageFolder.builder()
                .setRepositoryPath(Paths.get(folder))
                .addTransform(new Resize(224, 224))
                .addTransform(new ToTensor())
                .setSampling(8, true) // 8: the number of samples that are processed at once during each iteration of
                                      // training
                .build();
        dataset.prepare(new ProgressBar());
        return dataset;

    }
}
