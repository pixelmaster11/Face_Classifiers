# Face_Classifiers

This project is used for different types of classifications such as gender, ethnicity, etc attributes that can be extracted from face images. 
This project also provides very easy face detection, face recognition using Dlib and opencv libraries. 

**I.** To create a classifier whether it is for gender or ethnicity or face recognition or a custom one, the image directory should be as follows:

Image Directory

    |__ Face_Recognition
    │       ├── Bruce Lee
    │       ├── Mr Bean
    │       ├── Fernando Torres
    │       ├── Mo Salah
    │       ├── Sadio Mane
    │   
    |   
    |__ Gender_Recognition
    |       ├── Male
    |       ├── Female
    |   
    |   
    |__ Ethnicity_Recognition
    |       ├── Asian
    |       ├── Black
    |       ├── Indian
    |       ├── White


This structure is important as the label names are extracted from the respective Folder Names. Thus, for face recognition the labels for all images under folder Mo Salah will be given the label as Mo Salah and so on. 

**II.** The next step is to extract embeddings from the images which would be our feature vectors. These embeddings are computed using a pretrained CNN model from dlib library. These are 128-D embeddings. These embeddings will be extracted and stored alongwith their labels and this will be used as the inputs to our classifier models. 
    
    To generate the embeddings use the script generate_dataset.py as follows:
    
    python generate_dataset.py --image_dataset_dir="path/to/image_dir/" --embed_filename="somename_embeddings" --mode="save" 
    --embeddings_save_dir="path/where/generated/embeddings/file/will/be/saved"
    
   This will generate a .pkl file at your given path. This file stores embeddings, labels, image_paths in a single file.
   
**III.** Next step will be to train our classifier model using the generated embeddings and labels.
     There are couple of classifier classes provided such as svm.py or knn.py which allows you to optimize the hyper parameters. If you want to skip optimization, you can directly run classifier.py which uses a SVM classifier.
     
     To run the classifier use the script classifier.py as follows:
     
     python classifier.py --embeddings_load_dir="path/to/generated/embeddings/file" --embed_filename="name_of_embed_file_to_load"
     --mode="save" --model_filename="name/of/generated/classifier_model/to/save" --model_save_dir="path/where/models/willbe/saved"
     
   This will train a SVM classifier model by default on the given features and labels. 
   By default it will scale the features using Normalization and perform a Shuffle split cross validation with 10 splits with 
   train / test split as 70-30. It also performs a pure train-test 70-30 split without cross validation by training on the 70% train-      set and testing on the remaining 30% test-set and plots a confusion matrix with a classification report for detailed summary.   
   Once training is completed it will save the trained classifier model as a your_model_name.pkl file
   
    
