CSC 4240 Project -------------------------------------------------------------------------
- 

IMPORTANT!!!!!!!! ------------------------------------------------------------------------
1. This application utilizes torch, and as a result must use a python version between 
   3.9-3.12, otherwise it WILL NOT work (python 3.13 is NOT supported).

Things to install before running ---------------------------------------------------------
1. For this project, we utilized the free community edition pycharm IDE that is availabe
   for download from the following link: 
   https://www.jetbrains.com/pycharm/download/?section=windows

2. As stated above, you will also need a python version between 3.9 and 3.12 in order to be
   able to use torch. You can download the python version we used at the following link:
   https://www.python.org/downloads/release/python-3109/
   Make sure to add it to your path. You may need to restart for this to take effect

3. Once you have python downloaded added to your path, you can now use pip to install 
   some of the libraries we will be using. You will need to install the following:
   
   1. pip install openai==0.28
   2. pip install transformers
   3. pip install textblob
       - For this to work, you must install rust first before running this. You can do
         so by going to the following link: https://rustup.rs/

   4. pip3 install torch torchvision torchaudio --index-url 
      https://download.pytorch.org/whl/cu118
       - Note that this command utilizes CUDA as the compute platform. If you do not own
         a powerful GPU then head over to https://pytorch.org/ and configure pytorch for your specific system. It will give you a command line prompt to run.
