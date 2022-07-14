# Traffic-Sign-Detectio-App

The app takes in an image and runs a sliding window of varying sizes on the image. For each window, it runs the HoG feature extractor to get a feature vector which is then passed onto an SVM model for inference. If the model confidence is greater than 0.5 then save the bounding box coordinates in a list and once the entire image is traversed, draw bounding boxes for detected signs.
