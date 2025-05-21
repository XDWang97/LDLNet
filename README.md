# The Devil is in the Masks: Label Disentangling Learning Network for Camouflaged Object Detection



## 1. Preface

- This repository provides code for "The Devil is in the Masks: Label Disentangling Learning Network for Camouflaged Object Detection" 

## 2. Trainning and Testing

1. Configuring your environment (Prerequisites):
    
    + Creating a virtual environment in terminal: `conda create -n LDLNet python=3.9`.
    
    + Installing necessary packages: `pip install -r requirements.txt`.
2. Downloading necessary data:

    + downloading testing dataset and move it into `./data/TestDataset/`, 
    which can be found in this [download link (Google Drive)](https://drive.google.com/file/d/1SLRB5Wg1Hdy7CQ74s3mTQ3ChhjFRSFdZ/view?usp=sharing).
    
    + downloading training dataset and move it into `./data/TrainDataset/`, 
    which can be found in this [download link (Google Drive)](https://drive.google.com/file/d/1Kifp7I0n9dlWKXXNIbN7kgyokoRY4Yz7/view?usp=sharing).
3. Downloading necessary weights:
   + downloading smt_small weights and move it into `./models/smt_small.pth`[download link (github)](https://github.com/AFeng-x/SMT/releases/download/v1.0.0/smt_small.pth).
4. Next, for training and testing run`train.sh`.

## 3. Our Results.
  1. Our checkpoints can be found here [Google Drive](https://drive.google.com/file/d/1IhqsGC3a9ESFxHcbx9v7nvUm8hs6wbc_/view?usp=drive_link).
  2. Our results can be found here [Google Drive](https://drive.google.com/file/d/1XWBDifpOlWBmExTD6hP43hc7Xe2NFA8B/view?usp=drive_link).
