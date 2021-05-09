# CS205 Final Project (UNet for Nuclei Segmentation)
This is a repo for UNet implementation in CS205 final project. 
The primary infrastructure is Google Cloud Platform.
  


## Download data: 
Assume the dataset is saved in Google Cloud Storage bucket 'cs205_final_project'.
To run the code, first download the data to local machine by the following commands:
```
gsutil -m cp -R gs://cs205_final_project/cell_imgs data/
gsutil -m cp -R gs://cs205_final_project/mask_imgs data/
gsutil -m cp -R gs://cs205_final_project/test_imgs data/
```
Note: need ```gcloud init``` with the account that can access the bucket

## Setup environment:
```
python3 --version
```
If not python 3.6 then use the additional step ```sudo apt install python3.6```.
Then continue:
```
sudo apt update
sudo apt-get install software-properties-common
sudo apt-get install python3-pip
python3 -m pip install --upgrade pip setuptools
pip install -r requirements.txt
pip install google-cloud-profiler --ignore-installed
```
   
## Code Execution
To run the code, use the command 
```
python3 main.py --data_dir <path> --output_dir <path> --log_dir <path> --model_name <str> --epochs <num>
```
Flag descriptions:
- data_dir: directory that contains the datasets: cell_imgs, mask_imgs, test_imgs  
- log_dir: directory to save tensorboard logs
- output_dir: directory to save example prediction outputs
- epochs: number of training epochs  
- model_name: name of the model to be saved

For example, 
```
python3 main.py --root_dir /home/user/data --log_dir /home/user/logs --output_dir /home/user/outs --epochs 5 --model_name unet_ep5
```
## Code Profiling
We use cProfile and TensorBoard for code profiling. 

### cProfile
Use the following commands to run the main file with cProfile.
```
python3 -m cProfile -o main.profile main.py --root_dir xxx --output_dir xxx --epochs 5
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

### TensorBoard
We also use TensorBoard to visualize the model-specific profiling. 
To view the logs, first download the logs folder you specified after ```--log_dir``` to your local machine.
To download, assume you have set up Google Cloud SDK locally. Use the following command to download the logs 
to your current directory:
```
gcloud config set project <your-project-id>
gcloud compute scp --recurse your-instance:/dir/to/logs .
```
Then, make sure you have the correct version of Tensorboard installed by running the following:
```
pip uninstall tensorboard
pip install tensorboard==2.2.0
pip install -U tensorboard-plugin-profile==2.2.0
```
Then use the following command to launch tensorboard:
```
tensorboard --logdir dir/to/log
>> TensorBoard 2.2.0 at http://localhost:6006/ (Press CTRL+C to quit)
```
TensorBoard should now be running at ```http://localhost:6006/```. 
Paste the url in Chrome to visualize the execution graphs.
  
For profile information, select the PROFILE tab in the orange header bar and you can see a detailed
breakdown of the time spent in different sections. More detailed information can be found in
the 'Tools' drop-down list at the left.