import os
import cv2
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from matplotlib import pyplot as plt
from functools import reduce
import numpy as np

#----------Reading images from respective directory (./Faces)----------
def get_data():
    data = []
    labels = []
    class_id = {}
    count = 0
    for folder in os.listdir("./Faces"):
        if folder.startswith('.'):
            continue
        
        path_to_folder = os.path.join("./Faces", folder)
        class_id[count] = folder        
        
        for file in os.listdir(path_to_folder):
            if file.startswith('.'):
                continue
            
            path_to_file = os.path.join(path_to_folder, file)
            image = cv2.imread(path_to_file, cv2.IMREAD_GRAYSCALE)
            data.append(image)
            labels.append(count)
        
        count+=1

    return np.array(data), np.array(labels), class_id;

#----------Flatten images into 1 dimension----------
def ravel_data(data):
    data_ravel = []
    for image in data:
        data_ravel.append(image.ravel())
    return np.array(data_ravel)

#----------PCA Face Recognition----------
def pca(data, labels, n_components):
    #----------Applying PCA to obtain eigenfaces----------
    data_ravel = ravel_data(data)
    pca = PCA(n_components = n_components, whiten= True)
    pca.fit(data_ravel)
    pca_data = pca.transform(data_ravel)
    
    total = 0
    top_1_accuracy = 0
    top_3_accuracy = 0 
    top_10_accuracy = 0
    
    #----------Classification models to be used for fitting and prediciton----------
    models = []
    models.append(LogisticRegression(solver='lbfgs', multi_class='auto'))
    models.append(SVC(kernel='linear'))
    models[1].probability = True

    for model in models:
        train_data, test_data, train_labels, test_labels = train_test_split(pca_data, labels, test_size = 0.2, random_state = 100, shuffle = True)
        model.fit(train_data, train_labels)
        
        prediction = model.predict_proba(test_data)
        
        for i in range(len(prediction)):
            total += 1

            pred = (-prediction[i]).argsort()[:1]
            for p in pred:
                if p == test_labels[i]:
                    top_1_accuracy += 1
            
            pred = (-prediction[i]).argsort()[:3]
            for p in pred:
                if p == test_labels[i]:
                    top_3_accuracy += 1
                    break


            pred = (-prediction[i]).argsort()[:10]
            for p in pred:
                if p == test_labels[i]:
                    top_10_accuracy += 1
                    break

        print("\nPCA (n_components =", n_components ,") with", model.__class__.__name__)

        print("Top 1 accuracy", top_1_accuracy/total)
        print("Top 3 accuracy", top_3_accuracy/total)
        print("Top 10 accuracy", top_10_accuracy/total)

#----------Testing LDA and LBP models----------
def test(data, labels, model, class_id):
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size = 0.2, random_state = 100, shuffle = True)
    total = 0
    top_1_accuracy = 0
    top_3_accuracy = 0 
    top_10_accuracy = 0
    
    model.train(train_data, train_labels)
    predictions = cv2.face.StandardCollector_create()
    
    for i in range(len(test_data)):
        id = model.predict(test_data[i])[0]
        model.predict_collect(test_data[i], predictions)
        total += 1

        results = predictions.getResults(sorted = True)

        num = 0
        for (label, dist) in results:
            if num < 1:
                if label == test_labels[i]:
                    top_1_accuracy += 1
                    break
            else:
                break
            num += 1

        num = 0
        for (label, dist) in results:
            if num < 3:
                if label == test_labels[i]:
                    top_3_accuracy += 1
                    break
            else:
                break
            num += 1

        num = 0
        for (label, dist) in results:
            if num < 10:
                if label == test_labels[i]:
                    top_10_accuracy += 1
                    break
            else: 
                break
            num += 1
    
    print("Top 1 accuracy", top_1_accuracy/total)
    print("Top 3 accuracy", top_3_accuracy/total)
    print("Top 10 accuracy", top_10_accuracy/total)

data,labels,class_id = get_data()

pca(data, labels, 10)

pca(data, labels, 19)

pca(data, labels, 50)

pca(data, labels, 100)

pca(data, labels, 150)

pca(data, labels, 200)

print("\nLDA")

model = cv2.face.FisherFaceRecognizer_create()
test(data, labels, model, class_id)

print("\nLBP")

model = cv2.face.LBPHFaceRecognizer_create()
test(data, labels, model, class_id)
