import sys
import numpy as np

def main():
    argcnt = len(sys.argv)
    print("Number of arguments:", argcnt)
    
    for i in range(argcnt):
        print(i, ".", sys.argv[i])
        
    for a in sys.argv:
        print("-", a)
        
    all_frames = []
    
    for i in range(5):
        image = np.zeros((480,640,3), dtype="uint8")
        all_frames.append(image)
        
    cnt = len(all_frames)
    
        

if __name__ == "__main__":
    main()