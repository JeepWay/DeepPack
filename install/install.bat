@echo off
call D:\anaconda3\Scripts\activate.bat
call conda deactivate
call conda activate base 
echo Y | conda create -n deeppack python=3.9.15
call conda activate deeppack
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install matplotlib==3.8.2
pip install scipy==1.13.0
pip install tqdm==4.66.1
pip install imageio==2.34.1