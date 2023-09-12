### Docker pLGG_Segmentation
Main repo: https://github.com/AIM-KannLab/pLGG_Segmentation

Prerequisites: 
- Install docker: https://docs.docker.com/engine/install
- Clone repo: `git clone` 
- GPU: Nvidia GPU with CUDA support (tested on Nvidia A6000)
- The pretrained models can be downloaded at the following Drive link https://drive.google.com/file/d/1cbi3p9IoKWjKR-pl3yXde6ISx4hZy2DB/view?usp=sharing. Unzip this file, the unzipped folder should be named nnUNet_trained_models and placed in 'nnUnet/' folder.

## How to run the docker image inference on single MRI T2w image
0. Put all images in the 'example_input' folder
1. Build the docker image: 
`sudo docker build -t segmentation .`
*Note*: this will take a while to build the docker image
2. Run the docker image: 
`sudo docker run -i --gpus=all --ipc=host -v ${PWD}/output:/output/preprocessed -t segmentation` 
*Note*: this will take a while to run the docker image, depending on how many images you have in the 'example_input' folder/ how many GPUs you have. Your output will be in the 'preprocessed' folder.

!To pass parameters to docker(modify input/output paths), when running the docker image, use the following command: 
`sudo docker run -i --gpus=all --ipc=host -v ${PWD}/output:./output/ -t segmentation --CUDA_VISIBLE_DEVICES=0 --T2W_dir /example_input --output_path /output/`

*Optional:* to clean dockers `sudo docker image prune -a` or `docker image prune -a --filter "until=24h"` or `docker system prune`

## Dataset
Example dataset (note, this is not the example MRI with brain tumor; this is just a sample dataset to test the pipeline):
https://openneuro.org/datasets/ds000228/versions/1.1.0 
To run on your own data, place T2w .nii.gz files in the 'example_input' folder.

