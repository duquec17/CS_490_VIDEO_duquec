import requests 
import os
import pandas as pd
import shutil
import zipfile
from pathlib import Path

def download_file(file_url, output_dir, output_filename):   
  
    r = requests.get(file_url, stream=True, verify=True)     
    save_path = os.path.join(output_dir, output_filename)
    
    with open(save_path,'wb') as f:          
        for chunk in r.iter_content(chunk_size=1024):
            if chunk: 
                f.write(chunk) 
                
def unzip_file(zip_file, out_folder):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(out_folder)

def main():    
    dog_url = "https://huggingface.co/datasets/l-lt/LaSOT/resolve/main/dog.zip"
    zip_filename = "dog.zip"
    data_dir = os.path.join(".", "data")
    os.makedirs(data_dir, exist_ok=True)
    download_file(dog_url, data_dir, zip_filename)
    out_data_dir = os.path.join(data_dir, Path(zip_filename).stem)
    
    full_zip_path = os.path.join(data_dir, zip_filename)
    
    unzip_file(full_zip_path, out_data_dir)
    
    os.remove(os.path.join(data_dir, zip_filename))
    
if __name__ == "__main__":
    main()
    