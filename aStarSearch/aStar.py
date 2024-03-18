
"""
Name: Max Shooster
File: aStar.py

Description: Given a 395x500 image representing a terrain map where each pixel
corresponds to an area of 10.29m in longitude and 7.55m in latitude, and given
a text file of files corresponding to the elevation of each pixel, and given a 
text file with two integers per line representing the (x,y) points to visit, 
output the optimal path to complete the course in the shortest time possible.
The program uses an A* implementation to find the shortest optimal path. Then,
it outputs an image with the optimal path drawn on top of it, and it outputs 
the total distance of the path in meters to the console.

"""

import os
import sys
import heapq
import numpy as np
from PIL import Image
import math

# terrain penalties 
terrainPenalties = {
    (248, 148, 18): 1,      # Open land
    (0, 0, 0): 2,           # Footpath
    (71, 51, 3): 3,         # Paved road
    (255, 255, 255): 4,     # Easy movement forest
    (255, 192, 0): 5,       # Rough meadow
    (2, 208, 60): 6,        # Slow run forest
    (2, 136, 40): 7,        # Walk forest
    (5, 73, 24): 75,        # Impassible vegetation
    (0, 0, 255): 100,       # Lake/Swamp/Marsh
    (205, 0, 101): 1000     # Out of bounds
}

# Scaling Factors (in meters per pixel)
scales = {
    'x': 10.29,
    'y': 7.55
}

# 3D Heuristic function for A* (Calculates distance travelled, considers elevation)
def calculate3DHeuristic(a, b, elevationVals):
    scaleX = scales['x']
    scaleY = scales['y']
    # Calculate 2D distance difference using scale factor
    dx = abs(a[0]-b[0]) * scaleX
    dy = abs(a[1]-b[1]) * scaleY
    # Calculate elevation difference
    dz = abs((elevationVals[a[1], a[0]]) - (elevationVals[b[1], b[0]]))
    # Calculate 3D distance
    dist = math.sqrt(math.pow(dx, 2) + math.pow(dy, 2) + math.pow(dz, 2))
    return dist

# Function to get the terrain penalty
def getTerrainPenalty(b, terrainNP):
    terrainPenalty = terrainPenalties.get(tuple(terrainNP[b[1], b[0], :3]))
    return terrainPenalty

# Function to output optimal A* path
def backtrackPath(prev, curr):
    path = [curr]
    while curr in prev:
        curr = prev[curr]
        path.append(curr)
    #return the reverse order
    return path[::-1]

# A* search algorithm
def aStarSearch(beg, end, terrainNP, elevationVals):
    # Initialize priority queue
    priorityQueue = []
    # Used to keep track of an optimal path
    prev = {}
    heapq.heappush(priorityQueue, (0, beg)) #(priority,node)
    # Cost from start to node
    gScore = {beg: 0}
    # Cost from start to end
    fScore = {beg: calculate3DHeuristic(beg, end, elevationVals)}
    directions = [(0, 1), (0, -1), (-1, 0), (1, 0)] # N S W E
    
    while priorityQueue:
        # Pop node with lowest f score
        tmpVal = heapq.heappop(priorityQueue)
        current = tmpVal[1] # xy-coords
        # If goal reached, output optimal path
        if current == end:
            return backtrackPath(prev, current)
        # Explore neighbors
        for dx, dy in directions:
            neighbor = (current[0] + dx, current[1] + dy)
            # Check if neighbor is within image dimensions
            if (0 <= neighbor[0] < elevationVals.shape[1]) and (0 <= neighbor[1] < elevationVals.shape[0]):
                # Update temp g score of neighbor
                tmpGScore = gScore[current] + (calculate3DHeuristic(current, neighbor, elevationVals) * getTerrainPenalty(neighbor, terrainNP))
                # If path to neighbor is better, update
                if (neighbor not in gScore) or (tmpGScore < gScore[neighbor]):
                    prev[neighbor] = current
                    gScore[neighbor] = tmpGScore
                    fScore[neighbor] = gScore[neighbor] + calculate3DHeuristic(neighbor, end, elevationVals)
                    heapq.heappush(priorityQueue, (fScore[neighbor], neighbor))
    # If goal never reached, return an empty path
    return []

# Load in elevation data, disregard the last 5 values of each line
def parseElevationVals(filename):
    with open(filename, "r") as file:
        elevationVals = []
        for line in file:
            elevationVals.append([float(val) for val in line.split()[:-5]])
    return np.array(elevationVals)

# Load in points to visit
def parsePoints(filename):
    with open(filename, "r") as file:
        points = []
        for line in file:
            point = tuple(int(val) for val in line.split())
            points.append(point)
    return points

def main(args):
    # Load image
    terrainImg = Image.open(args[0])
    terrainNP = np.array(terrainImg)
    # Load elevation and points values
    elevationVals = parseElevationVals(args[1])
    pointsToVisit = parsePoints(args[2])
    # Total distance travelled (in meters)
    totalDist = 0.0
    # Store optimal path
    pathOptimal = set()
    # run A* search on each pair of points to find optimal path
    for i in range(len(pointsToVisit) - 1):
        singlePath = aStarSearch(pointsToVisit[i], pointsToVisit[i+1], terrainNP, elevationVals)
        # calculate total distance of path
        for j in range(1, len(singlePath)):
            totalDist += calculate3DHeuristic(singlePath[j], singlePath[j-1], elevationVals)
        pathOptimal.update(singlePath)
    # Draw optimal path on input image
    isRgba = terrainNP.shape[-1] == 4 # verifies if has an alpha channel
    for pixel in pathOptimal: # for each (x,y) tuple in the list of tuples
        if isRgba:
            # If image is RGBA, assign the full designated RGBA value
            terrainNP[pixel[1], pixel[0]] = [118, 63, 231, 255]
        else:
            # If image is RGB, assign only the designated RGB values
            terrainNP[pixel[1], pixel[0], :3] = [118, 63, 231]
    # Save output image
    outputImg = Image.fromarray(terrainNP)
    outputImg.save(args[3])
    # Print total distance to the console
    print(totalDist)
    return 0

if __name__ == "__main__":
    main(sys.argv[1:])
