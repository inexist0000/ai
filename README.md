wget https://repo.anaconda.com/archive/Anaconda3-2024.06-1-Linux-x86_64.sh
bash Anaconda3-2024.06-1-Linux-x86_64.sh
source ~/anaconda3/bin/activate

echo ". /root/anaconda3/etc/profile.d/conda.sh" >> ~/.bashrc
source ~/.bashrc
eval "$(/root/anaconda3/bin/conda shell.bash hook)"

conda update pip
conda install python=3.12.4


conda create --name ai python=3.12.4
conda activate projectA
conda install --file requirements.txt

pip install -r requirements.txt


pip show transformers
pip show tokenizers
script -f /home/sinishiyeo/AI/terminal_log.txt

nohup python test.py > test.log 2>&1 &# ai
