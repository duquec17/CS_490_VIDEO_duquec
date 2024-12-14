import requests 
import os
from unrar import rarfile
from shutil import rmtree, copyfile
import glob
from pathlib import Path

# WARNING: Must install unrar
# conda install conda-forge::unrar
# If you get an error saying "Couldn't find path to unrar library",
#   then have to install the UnRAR dll/lib.
# WINDOWS:
# - Go to WinRAR add-on/extra page: https://www.rarlab.com/rar_add.htm
# - Download and run the UnRAR.dll installer: https://www.rarlab.com/rar/unrardll-710b2.exe
# - Create a system environment variable:
#       - Variable name: UNRAR_LIB_PATH
#       - Variable value: C:\Program Files (x86)\UnrarDLL\x64\UnRAR64.dll
# - This assumes that UnRAR was installed to that directory.
# - Close Visual Code and reopen.  Kill any open terminals in there as well once it reopens.
# LINUX/MAC:
# - The implication from this page (https://pypi.org/project/unrar/) is that you will 
#       have to download the source and compile it.  I suspect you will also have to set
#       an environment variable to point to the installed path.  
# 
# In the WORST case:
# - Comment out the import to unrar (line 3)
# - In prepare_data ONLY:
#       * Comment out the rmtree call (line 117)
#       * Comment out any references to extractall (lines 141, 144, 164)
#       * Comment out any calls to os.remove() (lines 167, 172, 173)
# - Run the script (it will fail)
# - Manually unrar the files and make sure that only desired classes remain
#       * Undergraduate: ["run", "walk"]
#       * Graduate: ["climb_stairs", "fall_floor", "run", "walk"]
# - Run the script again

# WARNING: If you want your data put somewhere OTHER than in the project directory,
# change this variable
CORE_DATA_DIR = os.path.join("..", "..", "Data")

def get_data_params(is_grad):
    if is_grad:
        suffix = "_G"
        print("Getting graduate data params...")
    else:
        suffix = "_UG"
        print("Getting undergraduate data params...")
        
    data_params = {}
    data_params["root_dir"] = os.path.join(CORE_DATA_DIR, "HMDB51" + suffix)
    data_params["split_dir"] = os.path.join(data_params["root_dir"], "splits")
    data_params["video_dir"] = os.path.join(data_params["root_dir"], "videos")
    
    if is_grad:
        data_params["class_list"] = ["climb_stairs", "fall_floor", "run", "walk"]
    else:
        data_params["class_list"] = ["run", "walk"]
                
    data_params["output_dir"] = os.path.join(".", "assign04", "output" + suffix)

    return data_params

def ask_for_correct_data_params():
    ans = input("Undergraduate or graduate version? [U/G] ")
    
    if ans == "U":
        is_grad = False
    elif ans == "G":
        is_grad = True
    else:
        raise ValueError("Invalid answer " + ans + "; exiting...")
    
    return get_data_params(is_grad)  

def download_file(file_url, output_dir, output_filename, print_chunk_stride=None):   
  
    r = requests.get(file_url, stream=True, verify=True)     
    save_path = os.path.join(output_dir, output_filename)
    
    chunk_cnt = 0
    with open(save_path,'wb') as f:          
        for chunk in r.iter_content(chunk_size=1024):
            if chunk: 
                f.write(chunk) 
                chunk_cnt += 1
                if print_chunk_stride is not None and chunk_cnt % print_chunk_stride == 0:
                    print(".", end="")
                    
    if print_chunk_stride is not None:
        print("\nDownload complete:", output_filename)
        
def recreate_path(mypath):
    if os.path.exists(mypath):
        rmtree(mypath)
    os.makedirs(mypath, exist_ok=True)
    
def remove_split_files(split_dir, classes_to_keep):
    # Get complete list of files to keep
    files_to_keep = []
    for classname in classes_to_keep:
        files_to_keep += glob.glob(classname + "*.txt", root_dir=split_dir)
    
    # Remove everyone else
    for filename in os.listdir(split_dir):
        if filename not in files_to_keep:
            os.remove(os.path.join(split_dir, filename))        
                     
def prepare_data(data_params):   
    # Does data folder exist?
    if os.path.exists(data_params["root_dir"]):
        ans = input(data_params["root_dir"] + " folder exists: do you want to recreate it? [Y/n] ")
        if ans != "Y":
            print("Exiting...")
            return
        
        # Remove old folder
        rmtree(data_params["root_dir"])
                
    # Prepare data folder
    os.makedirs(data_params["root_dir"], exist_ok=True)
    
    # Download RAR file for video data
    print("Downloading files...")
    video_rar_basename = "hmdb51_org"
    video_rar_filename = video_rar_basename + ".rar"
    download_file("http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar", 
                  data_params["root_dir"], 
                  video_rar_filename,
                  print_chunk_stride=10000)
    
    # Download split file
    split_rar_basename = "test_train_splits"
    split_rar_filename = split_rar_basename + ".rar"
    download_file("http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/test_train_splits.rar",
                  data_params["root_dir"],
                  split_rar_filename,
                  print_chunk_stride=1000)
        
    # Unrar these files        
    video_rar = rarfile.RarFile(os.path.join(data_params["root_dir"], video_rar_filename))
    video_rar.extractall(data_params["video_dir"])
    
    split_rar = rarfile.RarFile(os.path.join(data_params["root_dir"], split_rar_filename))
    split_rar.extractall(data_params["root_dir"])
    
    # Rename split folder
    orig_split_folder = os.path.join(data_params["root_dir"], "testTrainMulti_7030_splits")
    os.rename(orig_split_folder, data_params["split_dir"])
     
    # Remove split files we don't need/want
    remove_split_files(data_params["split_dir"], classes_to_keep=data_params["class_list"])   
        
    # Go through and unrar each individual video file
    all_video_rars = os.listdir(data_params["video_dir"])
    for rar_filename in all_video_rars:
        # Get basename and full path
        basename = Path(rar_filename).stem
        full_path = os.path.join(data_params["video_dir"], rar_filename)
        
        # Is this in our class list?
        if basename in data_params["class_list"]:        
            print("\tUnpacking", rar_filename)
            one_video_rar = rarfile.RarFile(full_path)
            one_video_rar.extractall(data_params["video_dir"])
            
        # Remove the rar file either way
        os.remove(full_path) 
            
    print("All videos unpacked.")    
               
    # Remove rar files
    os.remove(os.path.join(data_params["root_dir"], video_rar_filename))
    os.remove(os.path.join(data_params["root_dir"], split_rar_filename))
    print("Cleanup complete.")
    
    print("Done!")
    
def main():    
    data_params = ask_for_correct_data_params()
    prepare_data(data_params)
    
if __name__ == "__main__":
    main()
    