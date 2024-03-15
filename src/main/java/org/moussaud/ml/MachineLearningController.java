package org.moussaud.ml;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class MachineLearningController {

     static Logger logger = LoggerFactory.getLogger(MachineLearningController.class);

    @Autowired
    MachineLearningService service;

    @GetMapping(value = "/check", produces = MediaType.APPLICATION_JSON_VALUE)
    public Results check() {

        Results results = new Results();
        results.add("data/test-1.png", service.check("data/test-1.png"));
        results.add("data/test-2.png", service.check("data/test-2.png"));
        results.add("data/test-3.png", service.check("data/test-3.png"));
        results.add("data/test-4.png", service.check("data/test-4.png"));

        return results;
    }

    @GetMapping(value = "/liveness")
	public String liveness() {
		logger.debug("liveness");
		return "okay";
	}

	@GetMapping(value = "/readiness")
	public String readiness() {
		logger.debug("readiness");
		return "okay";
	}

}
