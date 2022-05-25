### Table of contents
* [ACKNOWLEDGEMENT](#ACKNOWLEDGEMENT)
* [PIE dataset](#datasets)
* [Environment](#Environment_setup)
* [PreProcessing](#PreProcessing)
* [Trainining/Testing](#Training_and_Testing)
* [Refereces ](#Refereces)


<a name="ACKNOWLEDGEMENT"></a>
## ACKNOWLEDGEMENT
PIE preprocessing part is provided by PIE dataset creators but slight changes have been done to suit this task 
'data_preprocessing.py' is a combination of PIE dataset provided methods and custom methods for pre-processing

<a name="datasets"></a>
## PIE Dataset
For This task PIE dataset is used: 
The code is trained and tested with [Pedestrian Intention Estimation (PIE) dataset](http://data.nvision2.eecs.yorku.ca/PIE_dataset/).

Download annotations and video clips from the [PIE webpage](http://data.nvision2.eecs.yorku.ca/PIE_dataset/) and place them in the `PIE_dataset` directory. The folder structure should look like this:

```
PIE_dataset
    annotations
        set01
        set02
        ...
    annotations_attributes
        set01
        set02
        ...
    annotations_vehicle
        set01
        set02
        ...
    PIE_clips
        set01
        set02
        ...

```

Videos will be automatically split into individual frames for training. This will require **1.1T** of free space on the hard drive.

Create environment variables for PIE data root and add them to your `.bashrc`:

```
export PIE_PATH=/path/to/PIE/data/root
export PIE_RAW_PATH=/path/to/PIE/data/PIE_clips/

In this case:
export PIE_PATH=PIE_dataset
export PIE_RAW_PATH=PIE_dataset/PIE_clips
```


For training set01, set02 and Set04 were used for testing Set03 (videos 1-11) were used.
For the same setting as trained in my model you can ignore Set05 , Set06 and also Set03 (videos after 11)
If some videos are removed, their respective annotations should be removed (annotations,annotations_vehicle,annotations_attributes)
To set up datasets used for training / testing go to _get_image_set_ids in pie_data
You will find : image_set_nums = {'train': ['set01', 'set02', 'set04'],
                          'val': ['set05', 'set06'],
                          'test': ['set03'],
                          'all': ['set01', 'set02', 'set03',
                                  'set04', 'set05', 'set06']}
change according to the datasets you have 

<a name="Environment_setup"></a>
## Environment_setup 
If you have conda you can clone environment used for training as follows:
``` conda env create -f environment.yml ```
After creating environment, it can be activated by:
 ``` conda activate transformer_intent ```
If conda is not avilable you can install all dependencies from enironment.yml and requirements.txt manually


<a name="PreProcessing"></a>
## PreProcessing 
The first step is after downloading the Videos should be split into individual frames for training. 
To split videos into frames:  change the pie_path in image_extract to the path of the dataset... .../PIE_dataset then run:
``` python image_extract.py ```   Extracted images should be in folder PIE_dataset/images

In data_preprocessing main function change
``` data_path = path /to /PIE_dataset ```  

After images are extracted run data preprocessing -> this will create train,validation and test data 
``` python data_preprocessing.py ``` 
This will create filnames and labels for each type (training,test,validation)....filenames and labels can either be shuffled or normal


<a name="Training_and_Testing"></a>
## Training_and_Testing
To train and test the model run:
``` python transformer.py ``` 

In the main function (transformer.py)--> the type of data to be used can be selected such as shuffled or not shuffled
e.g for shuffled training set use train__filenames_shuffled and train_labels_shuffled the same can be done for validation and test
If shuffled sets are not created after data_preprocessing...go to data_preprocessing.py and change save_shuffled to True when calling image processing function

<a name="References"></a>
## References
* ** https://github.com/jeonsworld/ViT-pytorch/blob/main/train.py       VIT
* **https://medium.com/@mrgarg.rajat/training-on-large-datasets-that-dont-fit-in-memory-in-keras-60a974785d71  data preprocessing
* @inproceedings{rasouli2017they,
  title={PIE: A Large-Scale Dataset and Models for Pedestrian Intention Estimation and Trajectory Prediction},
  author={Rasouli, Amir and Kotseruba, Iuliia and Kunic, Toni and Tsotsos, John K},
  booktitle={ICCV},
  year={2019}
}
