# sudo apt update
sudo apt install python3 python3-dev python3-venv

wget https://bootstrap.pypa.io/get-pip.py
sudo python3 get-pip.py

pip --version

pip install -r requirements.txt

pip install google-cloud-profiler --ignore-installed
