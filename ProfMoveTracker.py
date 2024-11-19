from General_A03 import *
import skimage

def compute_optical_flow_farneback(video_frames):
    all_flow = []
    prev_frame = None
            
    for index, frame in enumerate(video_frames):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        print("** FRAME", index, "************************")
        if prev_frame is None:
            prev_frame = frame
         
          
        flow = cv2.calcOpticalFlowFarneback(prev_frame, frame,
                                            None, 0.5, 3, 
                                            winsize=15,
                                            iterations=3,
                                            poly_n=5, 
                                            poly_sigma=1.2, 
                                            flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
        
        #extra = np.zeros_like(flow[...,0])
        #combo = np.stack([flow[...,0], flow[...,1],extra], axis=-1)
                 
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        print(mag.shape, np.amin(mag), np.amax(mag))     
        mag /= 20.0
        combo = np.stack([mag, mag, mag], axis=-1)
        
        
        
                 
        all_flow.append(combo)
        prev_frame = frame
                
    return all_flow

def cluster_colors(image, k):
    samples = image.astype("float32")
    samples = np.reshape(samples, [-1, 3])
    # https://docs.opencv.org/4.x/d1/d5c/tutorial_py_kmeans_opencv.html
    ret, labels, centers = cv2.kmeans(samples, k, None, 
                                      (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 70, 0.1),
                                      3,cv2.KMEANS_RANDOM_CENTERS)
    #print(labels)
    #print(centers)
    recolor = centers[labels.flatten()]
    recolor = np.reshape(recolor, image.shape)
    #recolor /= 255.0
    recolor = cv2.convertScaleAbs(recolor)
    return recolor, labels, centers

def get_color_similarity(image, target):
    image = image - target
    image = image*image
    image = np.sum(image, axis=2, keepdims=True)
    image = np.sqrt(image)
    image /= 255.0
    #print(np.amax(image))
    image = 1.0 - image
    return image

def scale_box(box, psy, psx, ph, pw):
    (ymin, xmin, ymax, xmax) = box
    
    height = ymax - ymin
    width = xmax - xmin
    
    offy = psy*height
    offx = psx*width
    
    ymin += offy
    xmin += offx
    
    ymin = int(ymin)
    xmin = int(xmin)    
    
    height *= ph
    height = int(height)
    ymax = ymin + height
    
    
    width *= pw
    width = int(width)
    xmax = xmin + width
    
    return (ymin, xmin, ymax, xmax)

def get_hue_sat_histogram(subimage):    
    hsv = cv2.cvtColor(subimage, cv2.COLOR_BGR2HSV)    
    hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0,180,0,256])
    #cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
    hist /= np.sum(hist)
    return hist

def get_hue_val_histogram(subimage):    
    hsv = cv2.cvtColor(subimage, cv2.COLOR_BGR2HSV)    
    hist = cv2.calcHist([hsv], [0, 2], None, [180, 256], [0,180,0,256])
    #cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
    hist /= np.sum(hist)
    return hist
   
def main():
    # Load dog dataset
    dog_index = 7 #11
    max_images_to_load = 60 
    starting_index =  30 
    dog_images, dog_boxes, dog_video_name = load_dog_video(dog_index, 
                                                           max_images_to_load=max_images_to_load,
                                                           starting_frame_index=starting_index)
    
    #flow_frames = compute_optical_flow_farneback(dog_images)    
    
    fast = cv2.FastFeatureDetector_create()
    orb = cv2.ORB_create(nfeatures=1000)  
    
    cur_box = dog_boxes[0]    
    fokp = orb.detect(dog_images[0], None)         
    fokp, fodes = orb.compute(dog_images[0], fokp)
    
    bfmatch = cv2.BFMatcher(normType=cv2.NORM_HAMMING, crossCheck=True)
    
    
    # Loop through and show images
    index = 0
    key = -1
    ESC_KEY = 27
    while key != ESC_KEY:
        image = np.copy(dog_images[index])
        
        kp = fast.detect(image, None)
        okp = orb.detect(image, None)
        okp, odes = orb.compute(image, okp)
        print("ORB:", odes.shape)
        
        matches = bfmatch.match(fodes, odes)
                
        #print("KEYPOINTS:", kp)
        corner_image = np.copy(image)
        corner_image = cv2.drawKeypoints(corner_image, okp, None, color=(255,0,0))

        corner_image = cv2.drawKeypoints(corner_image, fokp, None, color=(0,0,255))

        src_pts = []
        dst_pts = []
        for match in matches:
            f_index = match.queryIdx
            t_index = match.trainIdx
            first = fokp[f_index]
            second = okp[t_index]       
            src_pts.append((int(first.pt[0]), int(first.pt[1])))
            dst_pts.append((int(second.pt[0]), int(second.pt[1])))   
            cv2.line(corner_image, 
                     (int(first.pt[0]), int(first.pt[1])), 
                     (int(second.pt[0]), int(second.pt[1])), (0,255,0), 2)
            
        src_pts = np.array(src_pts).astype("float32")
        dst_pts = np.array(dst_pts).astype("float32")
        transform, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        old_box = [
            [cur_box[1], cur_box[0]],
            [cur_box[3], cur_box[2]],
        ]
        old_box = np.array(old_box).astype("float32").reshape(-1,1,2)
        
        new_box = cv2.perspectiveTransform(old_box,transform)
        new_box = new_box.flatten().astype("int32")
        print(new_box)
        
        cur_box = (new_box[1], new_box[0], 
                   new_box[3], new_box[2])
        
        cv2.rectangle(corner_image, (new_box[0], new_box[1]),
                      (new_box[2], new_box[3]), (255,0,255), 3)
            
        fokp = okp
        fodes = odes
        
        (ymin, xmin, ymax, xmax) = dog_boxes[index]
        orig_subimage = image[ymin:ymax, xmin:xmax]
                
        slic_image = skimage.segmentation.slic(orig_subimage, 
                                               start_label=0)
        visual_slic = skimage.segmentation.mark_boundaries(orig_subimage, slic_image)
        
        number_superpixels = np.unique(slic_image)
        slic_subimage = np.copy(orig_subimage)
        #all_ave_colors = []
        for label in number_superpixels:
            coord = np.where(slic_image == label)
            mask = np.zeros(slic_image.shape[0:2], dtype="uint8")
            mask[coord] = 255
            ave_color = cv2.mean(orig_subimage, mask)[0:3]
            #all_ave_colors.append(ave_color)
            slic_subimage[coord] = ave_color
            #print(ave_color)
            
        #all_ave_colors = np.array(all_ave_colors)
                
        psy = 0.2
        psx = 0.0
        ph = 0.5
        pw = 1.0
        (symin, sxmin, symax, sxmax) = scale_box(dog_boxes[index],psy, psx, ph, pw)
        small_box = (symin, sxmin, symax, sxmax)
                
        alt_box = scale_box(small_box,-psy/ph,-psx/pw, 1.0/ph, 1.0/pw)
                
        draw_dog_box(image, alt_box, (0,0,0))
        draw_dog_box(image, dog_boxes[index], (0,255,0))
        
        subimage = image[symin:symax, sxmin:sxmax]
        k = 7
        cluster_image, labelmap, centers = cluster_colors(slic_subimage, k) #subimage, k)
        
        model_hue_sat = get_hue_sat_histogram(subimage)
        model_hue_val = get_hue_val_histogram(subimage)
        
        combo_model_hist = model_hue_sat + model_hue_val
        
        # Converting to unique index for hue-saturation
        max_index = 180*256
        hue_sat_index_hist = np.arange(max_index, dtype="float32")
        hue_sat_index_hist = np.reshape(hue_sat_index_hist, (180,256))
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        sub_hsv = cv2.cvtColor(subimage, cv2.COLOR_BGR2HSV)    
        img_hue_sat_index = cv2.calcBackProject([image_hsv], [0,1], 
                                                hue_sat_index_hist, [0,180,0,256],1)
        sub_hue_sat_index = cv2.calcBackProject([sub_hsv], [0,1], hue_sat_index_hist, [0,180,0,256],1)
        
        combo_image = np.stack([img_hue_sat_index, image_hsv[...,2]], axis=-1)
        combo_sub = np.stack([sub_hue_sat_index, sub_hsv[...,2]], axis=-1)
        
        sub_hist = cv2.calcHist([combo_sub], [0, 1], None, [max_index, 256], [0,max_index,0,256])
        cv2.normalize(sub_hist, sub_hist, 0, 255, cv2.NORM_MINMAX)
        
        heat_map_hsv = cv2.calcBackProject([combo_image], [0,1], sub_hist, [0,max_index,0,256],1)
        
        #print(heat_map_hsv.shape)
        #print(heat_map_hsv.dtype)
        
        
        labelmap = labelmap.flatten()
        
        max_index = 0
        max_cnt = 0
        for i in range(k):
            coords = np.where(labelmap == i)
            cnt = coords[0].shape[0]
            if cnt > max_cnt:
                max_cnt = cnt
                max_index = i
            #print(cnt)
        
        target = centers[max_index]
        
        heat_image = get_color_similarity(image, target)
             
        heat_image = np.where(heat_image > 0.7, heat_image, 0.0)
        
        
        cv2.imshow("DOG", image)
        cv2.imshow("SUBIMAGE", subimage)
        cv2.imshow("CLUSTER", cluster_image)
        cv2.imshow("HEAT", heat_image)
        cv2.imshow("SLIC", visual_slic)
        cv2.imshow("SLIC AVE", slic_subimage)
        cv2.imshow("HSV HEAT", heat_map_hsv)
        cv2.imshow("FASTDOG", corner_image)
        
        
        #cv2.imshow("FLOW", flow_frames[index])
        key = cv2.waitKey(-1) #33)
        
        index += 1
        if index >= len(dog_images):
            index = 0

if __name__ == "__main__":
    main()
    