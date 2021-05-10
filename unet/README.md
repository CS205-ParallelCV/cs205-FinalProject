# CS205 Final Project (UNet for Nuclei Segmentation)
This is a repo for UNet implementation in CS205 final project. 
The primary infrastructure is Google Cloud Platform.
  
  
First, set up Google Cloud SDK locally so that you can interact with the GC resources.
The set up documentation can be found on our website.  

Then instantiate a VM instance of type ```c2-standard-30 (30 vCPUs, 120 GB memory)``` with Ubuntu 18.04.



## Download data: 
Assume the dataset is saved in Google Cloud Storage bucket ```cs205_final_project```.
To run the code, first download the data to local machine by the following commands:
```
cd unet (there will be a directory called data inside)
gsutil -m cp -R gs://cs205_final_project/cell_imgs data/
gsutil -m cp -R gs://cs205_final_project/mask_imgs data/
gsutil -m cp -R gs://cs205_final_project/test_imgs data/
```
Note: need ```gcloud init``` with the account that can access the bucket


## Setup Environment:
Download this project and upload this ```unet``` directory to your VM instance with the following
command:
```
gcloud compute scp --recurse <dir/to/unet> <your-instance:/your/dir>
```
Or alternatively, do it manually by downloading the directory locally.
Then check Python version via
```
python3 --version
```
If it is not python 3.6, execute```sudo apt install python3.6``` to install it.  
Then continue:
```
sudo apt update
sudo apt-get install software-properties-common
sudo apt-get install python3-pip
python3 -m pip install --upgrade pip setuptools

chmod u+x setup.sh
./setup.sh

pip install -r requirements.txt
```

## Code Execution
To run the code, use the command 
```
python3 main.py --root_dir <path> --model_name <str> --epochs <num>
```
Flag descriptions:
- root_dir: the unet directory 
- epochs: number of training epochs  
- model_name: name of the model to be saved

For example, 
```
python3 main.py --root_dir /home/user/unet --model_name unet_ep5  --epochs 5
```
If you want to perform prediction on the test data with a pretrained model, use the following command
```bash
python3 main.py --root_dir /home/user/unet --weights_fp --test_dir /path/to/test_data --test_size [N] /path/to/your_weights.h5
```
Flag descriptions:
- root_dir: the unet directory 
- test_dir: directory to the test data  
- test_size: Number of images in test data folder  
- weights_fp: file path to the .h5 weights file


## Code Profiling
We use cProfile and TensorBoard for code profiling. 

### cProfile
Use the following commands to run the main file with cProfile.
```
python3 -m cProfile -o main.profile main.py --root_dir xxx --epochs 5
```
The code profile would be saved to the file 'main.profile'. To view its contents, run the following command:
```
python3 -m pstats main.profile

>> main.profile%
```
You will then enter an interactive mode. To view the top 10 most time-consuming functions, 
within the interactive mode, type the following commands after the percentage symbol in order:
```
main.profile%  strip
main.profile%  sort
main.profile%  sort time
main.profile%  stats 10
```
The following is an example output.
![](https://github.com/CS205-ParallelCV/cs205-FinalProject/blob/main/imgs/cProfile_output.jpg)

### TensorBoard
We also use TensorBoard to visualize the model-specific profiling.  
To view the logs, first download the ```logs``` folder in the ```results``` directory to your local machine.
Assuming you have set up Google Cloud SDK locally, you can use the following command to download the logs 
to your current directory:
```
gcloud config set project <your-project-id>
gcloud compute scp --recurse your-instance:/dir/to/logs .
```
Then, make sure you have the correct version of TensorBoard installed by running the following:
```
pip uninstall tensorboard
pip install tensorboard==2.2.0
pip install -U tensorboard-plugin-profile==2.2.0
```
Then use the following command to launch TensorBoard:
```
tensorboard --logdir=<dir/to/log>  

>> TensorBoard 2.2.0 at http://localhost:6006/ (Press CTRL+C to quit)
```
TensorBoard should now be running at ```http://localhost:6006/```. 
Paste the url in Google Chrome to visualize the execution graphs. 
(Don't use other browsers since profiling might not be available!)
  
For profile information, select the PROFILE tab in the orange header bar and you can see a detailed
breakdown of the time spent in different sections. More detailed information can be found in
the 'Tools' drop-down list at the left. The following is an example output.

![](https://github.com/CS205-ParallelCV/cs205-FinalProject/blob/main/imgs/Tensorboard_output.png)

## Execute with GPU on Google Cloud
To use the GPU for the compute engine instance on GCP, first make sure you have enough GPU quota in 
your account. Then follow the official guide to [attach](https://cloud.google.com/compute/docs/gpus/add-remove-gpus) 
and [install Cuda dependencies](https://www.tensorflow.org/install/gpu).  
After setup, make sure you ```nvndia-smi``` command works in the terminal.

Then, install tensorflow-gpu by 
```bash
pip install tensorflow-gpu
```

## Execute with GPU on AWS
To use GPU on AWS, we recommend using Deep Learning AMI (Ubuntu 18.04) Version 43.0 as your
the Amazon Machine Image, since it has most of the machine learning packages installed.  

First move the ```data.zip``` from our repo [here](https://github.com/CS205-ParallelCV/cs205-FinalProject/blob/main/data.zip)
 under this unet folder. Unzip it and install the package requirements.
```bash
tar -xvf data.zip
mv data unet/data
cd unet
pip3 install -r requirements.txt
pip3 install tensorflow-gpu
```
Make sure that when you finish setup, tensorflow is able to locate the GPU by running the following check.
```
python3
>>> import tensorflow as tf
>>> print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
>>> Num GPUs Available: 1
```
Finally, execute ```main.py``` to train UNet. 
```
python3 main.py --root_dir /home/ubuntu/unet --model_name unet_ep5 --epochs 5
```
Similarly, you can run ```main_predict.py``` to use pretrained model to predict on test data.
