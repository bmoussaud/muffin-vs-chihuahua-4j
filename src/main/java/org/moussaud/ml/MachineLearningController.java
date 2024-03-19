package org.moussaud.ml;

import java.net.URI;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.ModelAttribute;
import org.springframework.web.bind.annotation.PostMapping;

@Controller
public class MachineLearningController {

    static Logger logger = LoggerFactory.getLogger(MachineLearningController.class);

    @Autowired
    MachineLearningService service;

    @GetMapping(value = "/")
    String getForm(Model model) {
        model.addAttribute("check", new Results());
        return "/views/ml";
    }

   
    @GetMapping(value = "/check.html")
    String createNewsletter(@ModelAttribute CheckForm form, Model model) {
        model.addAttribute("check", check());
        return "/views/ml";
    }

    @GetMapping(value = "/check", produces = MediaType.APPLICATION_JSON_VALUE)
    public Results check() {

        Results results = new Results();
        results.add("/images/test-1.png", service.check("data/test-1.png"));
        results.add("/images/test-2.png", service.check("data/test-2.png"));
        results.add("/images/test-3.png", service.check("data/test-3.png"));
        results.add("/images/test-4.png", service.check("data/test-4.png"));

        return results;
    }

    @GetMapping(value = "/learn", produces = MediaType.APPLICATION_JSON_VALUE)
    public void learn() {
        service.learn(Constants.PATH_TO_TRAIN, Constants.PATH_TO_VALIDATE);
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

    class CheckForm {
        private Results results;

        @Override
        public String toString() {
            return "NewsletterForm [results=" + results + "]";
        }

        public Results getResults() {
            return results;
        }

        public void setResults(Results results) {
            this.results = results;
        }

    }

}
