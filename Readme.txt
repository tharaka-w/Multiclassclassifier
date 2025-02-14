Method 1

Use the python notebook directly and point the given .zip file and change following the way you unzip
    train_dir = '/content/data_progS24/data_progS24/train_processed/' # specify your training data directory
    train_anno_file = '/content/data_progS24/data_progS24/labels/train_anno.csv' # specify your training data label directory
    test_dir = '/content/data_progS24/data_progS24/test_processed/' # specify your test data directory
    test_anno_file = '/content/data_progS24/data_progS24/labels/test_anno.csv' # specify your test label directory

and run the model.

Last successful run is already saved in the python Notebook already to verify.


Method 2

Run preprocessing.py(Given by the instructor), and final_code.py separately by changing the same values mentioned above.

Change hyperparameters to your need.

Code is perfectly working and obtained a 95% accuracy in the last run which saved in the ipynb Notebook