# Live traffic jam detection

Current issues:
* ~~Model predictions are biased towards f rather than q due to there being
twice as many images for it in the training set~~ appears to be resolved with
better model?
* Training dataset has some differences from API data
* Use pretrained model to label API data and train on that?

Live traffic images from:
https://data.gov.sg/dataset/traffic-images

Training data can be found under 'trafficstate' from the following dataset:
https://github.com/corey-snyder/STREETS
