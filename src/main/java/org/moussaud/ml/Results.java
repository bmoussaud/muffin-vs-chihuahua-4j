package org.moussaud.ml;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.fasterxml.jackson.annotation.JsonProperty;

import ai.djl.modality.Classifications;
import ai.djl.modality.Classifications.Classification;

public class Results {

    static Logger logger = LoggerFactory.getLogger(Results.class);

    @JsonProperty(value = "result")

    List<Result> results = new ArrayList<>();

    public void add(String path, Classifications check) {
        logger.info("add path {}", path);
        var result = new Result();
        result.setUrl(path);
        List<Classification> topK = check.topK();
        for (Classification classification : topK) {
            var className = classification.getClassName();
            logger.info("className {}", className);
            if (className.compareToIgnoreCase("muffin") == 0) {
                result.setMuffin(classification.getProbability());
            }
            if (className.compareToIgnoreCase("chihuahua") == 0) {
                result.setChihuahua(classification.getProbability());
            }
        }
        logger.info("add new result {}", result);
        results.add(result);
    }

    public List<Result> getResults() {
        return results;
    }

}
