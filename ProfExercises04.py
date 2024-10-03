import cv2  
import numpy as np


def main():
    image = np.zeros((480,640,3), dtype="uint8")
    
    state = np.array([[100,100,3,3]], dtype="float64")
    
    A = np.array([[1,0,1,0],
                  [0,1,0,1],
                  [0,0,1,0],
                  [0,0,0,1]], dtype="float64")
    
    key = -1
    ESC_KEY = 27
    while key != ESC_KEY:
        image[:,:,:] = 0
        pos = (int(state[0,0]), int(state[0,1]))
        cv2.circle(image, pos, 4, (0,0,255), -1)
        
        cv2.imshow("IMAGE", image)
        key = cv2.waitKey(30)
        
        pred = np.transpose(np.matmul(A, np.transpose(state)))        
        state = pred    
        
        if state[0,0] < 0 or state[0,0] >= image.shape[1]:
            state[0,2] = -state[0,2]
            
        if state[0,1] < 0 or state[0,1] >= image.shape[0]:
            state[0,3] = -state[0,3]    

if __name__ == "__main__":
    main()
    
    