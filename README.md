# CS 490/548: Setup Instructions
***Author:* Dr. Michael J. Reale, SUNY Polytechnic**

## Miniconda Installation

Miniconda is a smaller version of Anaconda, which is a distribution of Python.  You will need to at least install Miniconda on a local machine (we will discuss how to make a portable version later in these instructions).

### Windows

First, download the latest installer for Windows [here](https://docs.conda.io/projects/miniconda/en/latest/).

Run the installer; I would install it for All Users (especially if your username has spaces in it).  I installed it into ```C:/miniconda3```.

Open "Anaconda Prompt (miniconda3)" **as an administrator**; you should see ```(base)``` to the left of your terminal prompt.

### Linux

First, download the latest version:
```
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
```

Next, install it into your home directory (default options are fine):
```
~/miniconda3/bin/conda init bash
```

Close and reopen the terminal; you should see ```(base)``` to the left of your terminal prompt.

### Mac
Open up a terminal.

If you are on an **Intel Mac**:
```
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
sh Miniconda3-latest-MacOSX-x86_64.sh
```
If you are on an **M1 Mac**:
```
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
sh Miniconda3-latest-MacOSX-arm64.sh
```
Close and reopen the terminal; you should see ```(base)``` to the left of your terminal prompt.

## Environment Creation
Create your CVvid environment with Python 3.10:
```
conda create -n CVvid python=3.10
```

Before installing any packages to the new environment, activate it:
```
conda activate CVvid
```

```(CVvid)``` should now be to the left of your terminal prompt.

## Installing PyTorch
With ```CVvid``` activated, you will need to install PyTorch for deep learning.  

### CUDA-Enabled PyTorch
If you have an NVIDIA graphics card (or intend to run this environment on a machine with one), you will want to install the CUDA-enabled version of PyTorch:

```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### CPU PyTorch
If you do NOT have an NVIDIA card (and/or are on a Mac), you will need to settle for the default installation:

```
pip3 install torch torchvision torchaudio
```

### Verifying PyTorch
To verify Pytorch works correctly:
```
python -c "import torch; x = torch.rand(5, 3); print(x); print(torch.cuda.is_available())"
```
You should see an array printed.  If you have an NVIDIA card, you should see the word ```True```.

## Installing Other Python Packages
With ```CVvid``` activated, run the following commands to install the necessary packages for this course:
```
pip3 install pandas scikit-learn scikit-image matplotlib 
pip3 install pylint gradio opencv-python jupyter
pip3 install accelerate 
pip3 install diffusers["torch"] transformers 
pip3 install --upgrade huggingface_hub
pip3 install huggingface_hub[cli,torch]
pip3 install peft
pip3 install --upgrade datasets
pip3 install bitsandbytes
pip3 install clean-fid
pip3 install ftfy regex tqdm
pip3 install av
pip3 install pytest
conda install -c conda-forge openh264
```

## Making a Portable Environment

While ```CVvid``` is active, install the ```conda-pack``` tool:
```
conda install -y -c conda-forge conda-pack
```

On Windows, I would recommend you ```cd``` to somewhere other than the default location (```C:\Windows\system32```).

Then, create your portable environment using conda-pack:
```
conda-pack --format zip
```
This will create ```CVvid.zip``` in whatever directory your terminal was in when you issued the command.

Create a folder on your USB drive (```CVvid```) and copy ```CVvid.zip``` to this folder.  Then, unzip the files "here".

***Linux***: **You MAY also have to manually copy ```etc\conda\activate.d\env_vars.sh``` to the corresponding folder on your USB drive!**

**PLEASE NOTE**: If the machine you are running this environment on does NOT have conda installed, you will NOT be able to install additional packages via conda (since the portable environment does not include the conda tool).

## Portable Visual Code
Go [here](https://code.visualstudio.com/Download) and download the **zip version** of Visual Code for your platform (most likely x64).
Unpack it to your USB drive.  Inside the folder for Visual Code, create a ```data``` folder:

This will cause Visual Code to store extensions and settings locally.

To run Visual Code:

***Windows:*** double-click on this version of ```Code.exe```.

***Linux/Mac:*** run ```./code```

Install the following extensions:
- **Python Extension Pack** by Don Jayamanne
- **Git Graph** by mhutchie
- **Pylint** by Microsoft

A terminal can always be created/opened with ```View menu -> Terminal```.  However, if you need to restart, click the garbage can icon on the terminal window to destroy it.

***Windows:*** Change your default terminal from Powershell to Command Prompt:
1. ```View menu -> Command Palette -> "Terminal: Select Default Profile"```
2. Choose ```"Command Prompt"```
3. Close any existing terminals in Visual Code

Once you open a project, to make sure you are using the correct Python interpreter:
1. Close any open terminals with the garbage can icon
2. Open a .py file
3. View -> Command Palette -> "Python: Select interpreter"
4. Choose the one located at ```CVvid\python.exe``` in your USB drive
5. If the GPU is not being utilized (or you see errors about paths to CUDA libraries not being found), type the following in your terminal to manually activate your environment: ```CVvid\Scripts\activate.bat``` (you will need whatever path is before ```CVvid```)

## Troubleshooting

* If you need to remove the CVvid environment (the LOCAL version, not the portable one):
```
conda remove --name CVvid --all
```

* If you encounter path issues (where conda isn't found once you activate an environment), open an Anaconda prompt as admin and try the following to add conda to your path globally: 
```
conda init
```

* If Intellisense is highlighting OpenCV (cv2) commands red:
    1. Open ```File menu -> Preferences -> Settings```
    2. Select the ```User``` settings tab
    3. Search for ```pylint```
    4. Under ```Pylint:Args```, add an item: ```--generate-members```

