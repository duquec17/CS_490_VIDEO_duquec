from General_A03 import *
from Test_A03 import *
import A03

GRAD_DOG_LIST = UG_DOG_LIST + [1,6,12,17,19,2]
#GRAD_DOG_LIST = [1,6,12,17,19,2]

def main():
    track_doggos(GRAD_DOG_LIST, show_dog_videos=show_dog_videos)
    
if __name__ == "__main__": 
    main()
    