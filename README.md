# CSC2516_Project
Github only have the code and the checkpoints cause github don't allow to upload files size larger than 100MB.
The code submit in markus will contain the testing data.
### Settings
Default setting is use Regression with skip connection models.
Can change to classification models in the following two files by setting args.classification = True or try the models with no skip connection by setting args.skip_connection = False.
    
    train_cifar.py
    train_lndscape.py
   
Each file is for one dataset.

### Evaluate with pretrained models
To evaluate the pre-trained models:

    python train_landscape.py
or

    python train_cifar.py

This will return the SSIM score.
