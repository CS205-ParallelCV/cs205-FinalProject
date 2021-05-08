# cs205-FinalProject
This is a repo for CS 205 final project.
  



download data: (need ```gcloud init``` with the account that can access the bucket)
```
gsutil cp -R gs://cs205_final_project/cell_imgs data
gsutil cp -R gs://cs205_final_project/mask_imgs data
gsutil cp -R gs://cs205_final_project/test_imgs data
```
setup environment:
```
python3 --version
sudo apt update
sudo apt-get install software-properties-common
sudo apt install python3.6
python3 --version
sudo apt-get install python3-pip
python3 -m pip install --upgrade pip setuptools
pip install -r requirements.txt
pip install google-cloud-profiler
```

To run the code, use the command ```python3 main.py --root_dir <path> --epochs <num> --weights <str>```
- root_dir: parent directory that contains cell_imgs, mask_imgs, test_imgs  
- epochs: number of training epochs  
- weights: name of the model to be saved, must end with '.h5'!  
For example, 
```
python3 main.py --root_dir /home/user/data --epochs 20 --weights model.h5
```
