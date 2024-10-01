import numpy as np
import math as m 
import cv2

def get_eigenvalues(a, b, d):
    apd = a + d
    
    under = apd*apd - 4*(a*d - b*b)
    root = m.sqrt(under)
    
    return (apd+root)/2.0, (apd-root)/2.0    

def get_eigenvector(lamb, a, b):
    y = -(a-lamb)/b
    length = m.sqrt(1 + y*y)    
    return [1/length, y/length]

def main():
    
    cnt = 100
    pts = np.random.normal((0,0), (40,20), (cnt, 2))
    color = np.random.normal(128, 50, (cnt,))
    print(pts.shape)

    angle = -45*m.pi/180.0
    R = np.array([[m.cos(angle), -m.sin(angle)],
                 [m.sin(angle), m.cos(angle)]], dtype="float64")
    
    pts = np.transpose(np.matmul(R, np.transpose(pts)))
    
    center = np.array([[320,240]], dtype="float64")
    pts = pts + center
    
    avepos = np.mean(pts, axis=0)
    print("AVE:", avepos)
    
    var_x = 0
    var_y = 0
    cov_xy = 0
    
    for p in pts:
        p = p - avepos
        var_x += p[0]*p[0]
        var_y += p[1]*p[1]
        cov_xy += p[0]*p[1]
    var_x /= cnt
    var_y /= cnt
    cov_xy /= cnt
    
    sigma = np.array([[var_x, cov_xy],
                      [cov_xy, var_y]], dtype="float64")
    
    
    lamb1, lamb2 = get_eigenvalues(var_x, cov_xy, var_y)
    print("LAMBDA:", lamb1, lamb2)
    
    vec1 = get_eigenvector(lamb1, var_x, cov_xy)
    print("EIGENVEC 1:", vec1)
        
    image = np.zeros((480,640,3), dtype="uint8")
    
    w_mean_pos = np.array([0,0], dtype="float64")
    sum_color = 0
    for i, p in enumerate(pts):
        one_color = int(255*(p[0] - 200)/400)
        w_mean_pos += one_color*p
        sum_color += one_color
        cv2.circle(image, p.astype(int), 3, (0,0,one_color), -1)
        
    w_mean_pos /= sum_color
    cv2.circle(image, w_mean_pos.astype(int), 5, (0,255,0), -1)
    
    cv2.imshow("IMAGE", image)
    cv2.waitKey(-1)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
    
