# introdution
this project is a sample project for evaluating ***Stable Deffussion(SD)*** with a dataset with chip defect.
note: this dataset is stored to private server, but you can use your owne dataset

**TRAINING METHOD**: in training SD the most important pice is data. you should gather high quality data and note size of images is efficient in generation size. you should just applay to model your target generation semantic (in my example defect is target so crop defects from complete images and applay just cropped defects)
**EVALUATION METHOD**: first step of evaluation is choosing metrices. after choosing metrices you should use trained model to evaluate.
**METRICES**: there is meny evaluation metrices but for SD or generative image models you have:
* CLIP score: established for measure an image or images to a text prompt
* IS score(Inception): established for measure quality of images. out puts of **IS** is mean and std and higher mean is better
* FID score(frechet inception distance): calculates the distance between fake or generated images and the real images. lower **FID** score represent higher quality.
* KID score (kernel inception distance):  measures the squared Maximum Mean Discrepancy (MMD) between the Inception representations of the real and generated samples using a polynomial kernel.in other word **KID** is improved **FID**. out puts of **KID** is mean and std and lower mean and std is better
# installation

        git clone https://github.com/Hamzeluie/SD_optimization.git
        dvc init
# stages
there is two stage in this project 
train and evalutaion