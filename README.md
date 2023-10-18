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
train and convert
* train: in this step. train model and evaluate. base on metrices wich you choose in 'eval_index' in params.yaml every 'checkpointing_steps'. if choosen metric tend to be muximize to be better results 'eval_index_max: True' else 'eval_index_max: False'.note 'resolution' parameter. it would result to size of output of SD.finally output of this stage is result/trainedmodel and reults/env/<dataset_name> you can see evaluation results in env folder. it contains generated fake images ,plots and metrics.
* convert: after training model.it convert pipelines to checkpoints.'checkpoint_path' is trained pipeline path and 'trained_model_path' is path of checkpoint to output ckpt files

# dvc notation
every training proccess and uploading dataset will track with dvc.
after while when you work with dvc repository you may change , delete and create dataset and trained model.it will take storage space even dataset that you removed and not currently use in the repository will be in '.dvc/catch' file.so you have heavy catch.which is not currently use. you can clean catch of dvc and remove the data that is not available currently by

        dvc gc --workspace

it will not remove catch of currently use data.just the data which is not use any more.
you can test it with this example.

        cd SD_optimization
        du -sh .dvc

check value of code then remove a dataset then 

        dvc gc --workspace
        du -sh .dvc

compare with past space.

