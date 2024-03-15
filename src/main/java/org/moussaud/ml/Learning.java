package org.moussaud.ml;

import java.io.IOException;
import java.io.Writer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

import ai.djl.Model;
import ai.djl.basicdataset.cv.classification.ImageFolder;
import ai.djl.basicmodelzoo.cv.classification.ResNetV1;
import ai.djl.metric.Metrics;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
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

public class Learning implements Constants {

    public static void main(String[] args) throws Exception {
        new Learning().run();
    }

    public void run() throws Exception {

        ImageFolder dataset = loadDataSet(PATH_TO_TRAIN);
        ImageFolder validationset = loadDataSet(PATH_TO_VALIDATE);

        try (Model model = getModel();
                Trainer trainer = model.newTrainer(getTrainingConfig())) {
            trainer.setMetrics(new Metrics());
            trainer.initialize(new Shape(1, 3, 244, 244));
            EasyTrain.fit(trainer, 1, dataset, validationset);

            Path modelDir = Paths.get("model");
            System.out.println("Save model " + model.getName());
            model.save(modelDir, model.getName());
            saveLabels(modelDir, dataset.getSynset());

        }
    }

    private  void saveLabels(Path modelDir, List<String> synset) throws IOException {
        var labelFile = modelDir.resolve("synset.txt");
        try (Writer writer = Files.newBufferedWriter(labelFile)) {
            writer.write(String.join("\n", synset));
        }
    }

    private  TrainingConfig getTrainingConfig() {
        return new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss())
                .addEvaluator(new Accuracy())
                .optExecutorService()
                .addTrainingListeners(TrainingListener.Defaults.logging(1));
    }

    private  ImageFolder loadDataSet(String folder) throws IOException {
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