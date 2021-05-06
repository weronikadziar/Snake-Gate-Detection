# Snake-Gate-Detection
Color and Edge Based Snake Gate Detection for Autonomous Drone Racing.
Algorithm written by Weronika Dziarnowska for the course "Autonomous Flight of Micro Air Vehicles" at Delft University of Technology.

The snake gate detection is based on:
S. Li, M.M.O.I. Ozo, C. De Wagter, and G.C.H.E.de Croon. Autonomous drone race: A computationally efficient vision-based navigation and control strategy. Robotics and Autonomous Systems, 133, 2020.

The algorithm is contained in one python file "Snake_Gate_Detection.py". The user should scroll down to the bottom of the code (around line 260) and input the correct filename and location of the image to be processed. 

The code can output the coordinates of the four corners if gate is found. The user should uncomment line 257 (end of last function) to output the coordinates. Another way of viewing the results is uncommenting the plotting commands (lines 242-251) within the last function to plot the color detection, edge detection, and corner detection. If the gate is detected, then the corners are marked with red dots. Otherwise, nothing is plotted over the image.
