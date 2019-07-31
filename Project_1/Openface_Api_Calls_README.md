==============================================================
Following are the api calls used in openface docker container
==============================================================

=========================================================
Training a classifier:
=========================================================

For N in {1..8}; do <openface-directory>/util/align-dlib.py <path-to-raw-data> align outerEyesAndNose <path-to-aligned-data> --size 96 & done

<openface-directory>/batch-represent/main.lua -outDir <feature-directory> -data <path-to-aligned-data>

<openface-directory>/demos/classifier.py train <feature-directory>

=========================================================
=========================================================
Classifying images using the trained classifier
=========================================================

<openface-directory>/demos/classifier.py infer <feature-directory>/classifier.pkl your_test_image.jpg

=========================================================
