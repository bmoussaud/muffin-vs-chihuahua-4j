package org.moussaud.ml;

import ai.djl.Model;
import ai.djl.basicmodelzoo.cv.classification.ResNetV1;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;

public interface Constants {
    static final String DATA_HOME = "/Users/bmoussaud/Downloads";
    static final String MUFFIN_VS_CHIHUAHUA_MODEL = "my-muffin-vs-chihuahua-model";
    static final String PATH_TO_TRAIN = DATA_HOME + "/muffin-vs-chihuahua/archive/train";
    static final String PATH_TO_VALIDATE = DATA_HOME + "/muffin-vs-chihuahua/archive/test";

    default Model getModel() {
        Model model = Model.newInstance(MUFFIN_VS_CHIHUAHUA_MODEL);
        Block resNetN50 = ResNetV1.builder()
                .setImageShape(new Shape(3, 224, 224))
                .setNumLayers(50)
                .setOutSize(2)
                .build();

        model.setBlock(resNetN50);
        return model;
    }
}
