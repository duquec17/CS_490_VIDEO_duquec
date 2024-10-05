import cv2  
import numpy as np


def main():
    
    logo = cv2.imread("dvdlogo.png")
    logo = 255 - logo
    logo = cv2.resize(logo, None, None, fx=0.1, fy=0.1)
    #cv2.imshow("LOGO", logo)
    
    def draw_logo(image, logo, x, y):
        rx = logo.shape[1]//2
        ry = logo.shape[0]//2
        sx = x - rx
        ex = sx + logo.shape[1]
        sy = y - ry
        ey = sy + logo.shape[0]
        image[sy:ey, sx:ex] = logo
        
    
    image = np.zeros((480,640,3), dtype="uint8")
    
    state = np.array([[100,100,3,3]], dtype="float64")
    
    A = np.array([[1,0,1,0],
                  [0,1,0,1],
                  [0,0,1,0],
                  [0,0,0,1]], dtype="float64")
    
    kalman = cv2.KalmanFilter(4, 2, type=cv2.CV_64F)
    kalman.measurementMatrix = np.array([[1,0,0,0],
                                        [0,1,0,0]], dtype="float64")
    kalman.transitionMatrix = A
    kalman.processNoiseCov = np.array([[1,0,0,0],
                                        [0,1,0,0],
                                        [0,0,1,0],
                                        [0,0,0,1]], dtype="float64")*1e-4
    kalman.measurementNoiseCov = np.array([[1,0],
                                        [0,1]], dtype="float64")*1e-1
    
    mouse_pos = [0,0]
    
    def mouse_func(action, x, y, flags, *userdata):       
        mouse_pos[0] = x
        mouse_pos[1] = y
        
    cv2.namedWindow("IMAGE")
    cv2.setMouseCallback("IMAGE", mouse_func)
     
    key = -1
    ESC_KEY = 27
    while key != ESC_KEY:
        image[:,:,:] = 0
        #state[0,0] = mouse_pos[0]
        #state[0,1] = mouse_pos[1]
        
        #draw_logo(image, logo, pos[0], pos[1])
        
        measurement = np.array([[state[0,0]],
                                 [state[0,1]]], dtype="float64")
        #measurement += np.random.normal(scale=10, size=(2,1))
        if np.random.randint(0,10) == 0:
            print("HELP!")
            measurement = np.array([[0],[0]], dtype="float64") #+= np.random.normal(scale=10000, size=(2,1))
        
        pos = (int(measurement[0,0]), int(measurement[1,0]))
        cv2.circle(image, pos, 4, (0,0,255), -1)
        
        pred = kalman.predict()
        kalman.correct(measurement)
        
        pred_pos = (int(pred[0,0]), int(pred[1,0]))
        cv2.circle(image, pred_pos, 3, (0,255,0), -1)
        
        print(pred.shape)
        
        cv2.imshow("IMAGE", image)
        key = cv2.waitKey(30)
        
        state = np.transpose(np.matmul(A, np.transpose(state)))        
        
        if state[0,0] < 0 or state[0,0] >= image.shape[1]:
            state[0,2] = -state[0,2]
            
        if state[0,1] < 0 or state[0,1] >= image.shape[0]:
            state[0,3] = -state[0,3]  
            
    cv2.destroyAllWindows()  

if __name__ == "__main__":
    main()
    
    