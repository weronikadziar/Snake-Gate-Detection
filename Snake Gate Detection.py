#------------ Color and Edge Based Snake Gate Detection -------------#
#------------------- For Autonomoous Drone Racing -------------------#
#--------------------- By Weronika Dziarnowska ----------------------#
# Developed for the course "Autonomous Flight of Micro Air Vehicles" #
#---------------- At Delft University of Technology -----------------#

# To run the code on an image, scroll all the way down and enter the 
# image name and path.

import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob

# Detect colors in the HSV range.
def color_filter(im):   

    lower_blue = np.array([90,50,95])
    upper_blue = np.array([140,255,255])
    filtered = cv2.inRange(im, lower_blue, upper_blue)

    return filtered

# Use Canny edge detection.
def edge_detector(im):
    return cv2.Canny(im,300,800)

# Classify image element as gate part or not. 
def detector(window,window_edges):
    
    # Detect the color of the gate within current image window.
    window_color = color_filter(window)
    # The pre-detected edges in the current window were already inputted.
    
    # Now count the pixels were blue is detected.
    color_nr = (window_color > 0).sum()
    # And count the pixels were edges are detected.
    edge_nr = (window_edges > 0).sum()
    
    # Accept current window as gate element if the pixels counted above are
    # above a certain threshold tuned for optimal performance.
    if (color_nr >= 0.3*len(window)**2) and (edge_nr >= 0.3*len(window)):
        detected = 1
    else:
        detected = 0
        
    return detected

# Detect gate elements in the entire image.
# This function is not used in the snake search algorithm, it was developed 
# to see how the combined color and edge detection work, and to tune the parameters.
# This function essentially outputs a mask that the snake search would produce
# if it searched through the entire image and not just along the edges.
# The inputs are the image, the pre-detected edges in the entire image, the
# kernel size that determines how big a window the detection procedure considers.
def color_edge(im,im_edges,kernel):
    size = len(im)
    # Kernel should be odd so find this "half" value for indexing.
    half = int((kernel-1)/2)
    # Start at the bottom.
    y = size - half
    # Pre-allocate a matrix that will be colored 1 if gate is detected and 0
    # if gate is not detected in the current window.
    detected_gate = np.zeros((size,size))
    # Run accross the entire vertical length.
    while (y - half >= 0):
        # Reset the x position to the far left.
        x = half
        # Run accross the entire horizontal length.
        while (x + half <= size):
            # Select a window of the image.
            window = im[y-half:y+half,x-half:x+half]
            # Select a window of the edge detection matrix.
            window_edges = im_edges[y-half:y+half,x-half:x+half]
            # Apply the detector function to determine if it's a gate.
            detected = detector(window,window_edges)
            if detected == 1:
                # Color the entire kernel as True.
                detected_gate[y-half:y+half,x-half:x+half] = 1
            # Move on to the next horizontal position.
            x = x + 1
        # Move up.
        y = y - 1
    return detected_gate

# This is the actual snake search main function.
# Use snake search to find the left vertical bar and the two horizontal bars.
def snake_gate_detection(im,im_edges,kernel,sigma):
    size = len(im)
    # Kernel should be odd so find this "half" value for indexing.
    half = int((kernel-1)/2)
    # Start at the bottom.
    y = size - half
    # Pre-allocate a value that is later used to determine if detection was successful.
    found = 0
    # Run until left vertical bar has been found.
    done = 0
    # Run vertically until done or until maximum image height is reached.
    while (done == 0) and (y - half >= 0):   
        # Reset x position to the far left.
        x = half
        # Run horizontally until done or until maximum image width is reached.
        while (done == 0) and (x + half <= size):        
            # Select current window of the image edge detection. This way, only
            # kernels with edges are even considered, which speeds up the search.
            window_edges = im_edges[y-half:y+half,x-half:x+half]           
            # If any edges were detected in this window, then check detected().
            if np.any(window_edges > 0):
                # Select current window of the image.
                window = im[y-half:y+half,x-half:x+half]
                # Check if conditions for color and edge detection are met.
                detected = detector(window,window_edges)
                if detected == 1:
                    # Set the current y and x as the position of the lower left 
                    # corner P1.
                    P1y = y
                    P1x = x
                    # Perform the search up to find the location of the upper 
                    # left corner P2.
                    [P2y,P2x] = search_up(y,x,im,im_edges,kernel)
                    # Check if the distance between P1 and P2 is sufficient.
                    if np.linalg.norm([P1y-P2y,P1x-P2x]) >= sigma:
                        # Perform the search to the right to find the lower
                        # right corner P3 and upper right corner P4.
                        [P3y,P3x] = search_right(P1y,P1x,im,im_edges,kernel)
                        [P4y,P4x] = search_right(P2y,P2x,im,im_edges,kernel)
                        # If any of the bars is long enough, then accept detection.
                        if (np.linalg.norm([P1y-P3y,P1x-P3x]) >= sigma) or (np.linalg.norm([P2y-P4y,P2x-P4x]) >= sigma):
                            done = 1
                            found = 1
            # If gate is not found yet, then keep looking for a better startin point.
            # Move to the right.
            x = x + half
        # Move up.
        y = y - half
    
    # If a gate was found, then refine the corners and output them
    if found == 1:
        points = np.array([[P1y,P1x],[P2y,P2x],[P3y,P3x],[P4y,P4x]])
        points = refine_corners(points,sigma)
        return points, found
    # Otherwise return found == 0. 
    else:
        return found, found
        
    
# Function to search up. It works according to the original algorithm but it
# also accepts x locations 2 pixels away, not just 1 pixel away.
def search_up(y,x,im,im_edges,kernel):
    done = 0
    size = len(im)
    half = int((kernel-1)/2)
    # Keep going until the end of the bar is reached or until maximum image
    # dimensions are exceeded.
    # The loop assesses if kernel up is a gate, allowing for going 1 or 2 
    # pixels to the right or to the left.
    while (done == 0) and (y-1-half >= 0) and (x-2-half>=0) and (x+2+half<=size):
        if detector(im[y-1-half:y-1+half,x-half:x+half],im_edges[y-1-half:y-1+half,x-half:x+half])==1:
            y = y - 1
        elif detector(im[y-1-half:y-1+half,x-1-half:x-1+half],im_edges[y-1-half:y-1+half,x-1-half:x-1+half])==1:
            y = y - 1 
            x = x - 1
        elif detector(im[y-1-half:y-1+half,x+1-half:x+1+half],im_edges[y-1-half:y-1+half,x+1-half:x+1+half])==1:
            y = y - 1 
            x = x + 1 
        elif detector(im[y-1-half:y-1+half,x-2-half:x-2+half],im_edges[y-1-half:y-1+half,x-2-half:x-2+half])==1:
            y = y - 1 
            x = x - 2
        elif detector(im[y-1-half:y-1+half,x+2-half:x+2+half],im_edges[y-1-half:y-1+half,x+2-half:x+2+half])==1:
            y = y - 1 
            x = x + 2 
        else:
            done = 1
    return y, x
        
    
# Function to search to the right. It works according to the original algorithm 
# but it also accepts y locations 2 pixels away, not just 1 pixel away.   
def search_right(y,x,im,im_edges,kernel):
    done = 0
    size = len(im)
    half = int((kernel-1)/2)
    # Keep going until the end of the bar is reached or until maximum image
    # dimensions are exceeded.
    # The loop assesses if the kernel to the right is a gate, allowing for
    # going 1 or 2 pixels up or down.
    while done == 0 and (y-1-half >= 0) and (y+1+half<=size) and (x+1+half<=size):
        if detector(im[y-half:y+half,x+1-half:x+1+half],im_edges[y-half:y+half,x+1-half:x+1+half])==1:
            x = x + 1
        elif detector(im[y+1-half:y+1+half,x+1-half:x+1+half],im_edges[y+1-half:y+1+half,x+1-half:x+1+half])==1:
            y = y + 1 
            x = x + 1 
        elif detector(im[y-1-half:y-1+half,x+1-half:x+1+half],im_edges[y-1-half:y-1+half,x+1-half:x+1+half])==1:
            y = y - 1 
            x = x + 1
        elif detector(im[y+2-half:y+2+half,x+1-half:x+1+half],im_edges[y+2-half:y+2+half,x+1-half:x+1+half])==1:
            y = y + 2 
            x = x + 1 
        elif detector(im[y-2-half:y-2+half,x+1-half:x+1+half],im_edges[y-2-half:y-2+half,x+1-half:x+1+half])==1:
            y = y - 2 
            x = x + 1
        else:
            done = 1 
    return y, x

# Function to refine the corners if one of the horizontal bars is too short.
def refine_corners(points,sigma):
    p = points
    if np.linalg.norm(p[2]-p[0]) < sigma:
        # If lower horizontal bar is too short, then construct point P3 using
        # the upper bar vector.
        p[2] = p[0] + (p[3]-p[1])
    elif np.linalg.norm(p[3]-p[1]) < sigma:
        # If upper horizontal bar is too short, then construct point P4 using
        # the lower bar vector.
        p[3] = p[1] + (p[2]-p[0])
    return points
        
   
# Run the snake search function and plot the results.   
def process_and_show_image(image):
    # Read the image.
    im = cv2.imread(image)
    # Convert the image to YUV
    im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    
    # Detect the gate (blue color), just for visualization, not for snake search.
    im_color = color_filter(im_hsv)
    
    # Detect the edges
    im_edges = edge_detector(im_hsv)
    
    # Create a gate mask based on color and edge detection, just for visualization,
    # not for snake search.
    # im_gate = color_edge(im_hsv, im_edges, 15)
    
    # Perform the snake search.
    [corners,found] = snake_gate_detection(im_hsv,im_edges,20,90)
    
    
    # # Uncomment the lines below to Plot
    # fig, axs = plt.subplots(1, 3, figsize=(35, 10))
    # axs[0].imshow(im_color, aspect="auto")
    # axs[1].imshow(im_edges, aspect="auto")
    # axs[2].imshow(cv2.cvtColor(im_hsv, cv2.COLOR_HSV2RGB), aspect="auto")
    # if found == 1:
    #     # Plot detection results, otherwise no points are plotted.
    #     axs[2].scatter(corners[0][1],corners[0][0],s=500,c='red')
    #     axs[2].scatter(corners[1][1],corners[1][0],s=500,c='red')
    #     axs[2].scatter(corners[2][1],corners[2][0],s=500,c='red')
    #     axs[2].scatter(corners[3][1],corners[3][0],s=500,c='red')
    
    # # To save the figures in a folder, uncomment the two lines below.
    # # number = image[11:-1]
    # # fig.savefig("Plots_ColorTune/snake_"+number+"g")
    
    # plt.show()
    
    # return corners


#------------------ Enter your image name and path here ------------------#
#-- Uncomment and use the commands for multiple or only one image below --#

# Detect gates in all images from the folder.
# for image in glob.glob("Images/*.png"): 
#     process_and_show_image(image)
    
# Or run the code just for one image.
# image = "Images/img_42.png"
# process_and_show_image(image)
    



    
