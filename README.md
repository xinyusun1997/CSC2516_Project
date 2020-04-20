# CSC2516_Project
Github only have the code and the checkpoints cause github doesn't allow to upload files size larger than 100MB.

### Dependence
pytorch, keras, opencv, PIL, skimage, sklearn, numpy, tqdm

Train and test with CUDA, haven't tested on CPU.
## Below requires the test data
Test data can be downloaded from 

https://drive.google.com/file/d/1MpkVXZvtxkKfn0HB33FKk8eiQ-M7rWIa/view?usp=sharing 

and put the ./data folder into the root place.
### Settings
Default setting is use Regression with skip connection models.
Can change to classification models in the following two files by setting args.classification = True or try the models with no skip connection by setting args.skip_connection = False.
    
    train_cifar.py
    train_lndscape.py
   
Each file is for one dataset.

### Evaluate with pretrained models
To evaluate the pre-trained models. Default is (regression with skip connection) model.

    python train_landscape.py
or

    python train_cifar.py

This will return the SSIM score. 
