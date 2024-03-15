package org.moussaud.ml;

import java.util.HashMap;
import java.util.Map;

import org.springframework.stereotype.Service;

import com.fasterxml.jackson.annotation.JsonProperty;

import ai.djl.modality.Classifications;


public class Results {

    @JsonProperty(value = "result")
    Map<String, String> results = new HashMap<>();

    public void add(String path, Classifications check) {
        results.put(path, check.toJson());
    }

}
