Description: Given a 395x500 image representing a terrain map where each pixel
corresponds to an area of 10.29m in longitude and 7.55m in latitude, and given
a text file of files corresponding to the elevation of each pixel, and given a 
text file with two integers per line representing the (x,y) points to visit, 
output the optimal path to complete the course in the shortest time possible.
The program uses an A* implementation to find the shortest optimal path. Then,
it outputs an image with the optimal path drawn on top of it, and it outputs 
the total distance of the path in meters to the console.
