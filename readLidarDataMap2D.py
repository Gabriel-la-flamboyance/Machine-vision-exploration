import math
from collections import deque

import matplotlib.pyplot as plt
import numpy as np

# Constants
EXTEND_AREA = 1.0  # Area extension around the grid map

def file_read(f):
    """
    Function to read angles and distances from a file.
    Args:
    f: Path to the file.
    
    Returns:
    angles: Numpy array of angles.
    distances: Numpy array of distances.
    """
    with open(f) as data:
        measures = [line.split(",") for line in data]
    angles = []
    distances = []
    for measure in measures:
        angles.append(float(measure[0]))
        distances.append(float(measure[1]))
    angles = np.array(angles)
    distances = np.array(distances)
    return angles, distances

def bresenham(start, end):
    """
    Bresenham's line algorithm to generate points between two coordinates.
    Args:
    start: Starting point (x1, y1).
    end: Ending point (x2, y2).
    
    Returns:
    points: Numpy array of points forming the line.
    """
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
    is_steep = abs(dy) > abs(dx)  # Check if the line is steep
    if is_steep:  # If steep, swap the coordinates
        x1, y1 = y1, x1
        x2, y2 = y2, x2
    
    swapped = False
    if x1 > x2:  # Ensure the line goes from left to right
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True
    dx = x2 - x1  
    dy = y2 - y1  
    error = int(dx / 2.0)  # Initialize the error term
    y_step = 1 if y1 < y2 else -1  # Determine y direction
    
    y = y1
    points = []
    for x in range(x1, x2 + 1):
        coord = [y, x] if is_steep else (x, y)  # Swap back if steep
        points.append(coord)
        error -= abs(dy)
        if error < 0:
            y += y_step
            error += dx
    if swapped:  # If points were swapped, reverse the list
        points.reverse()
    points = np.array(points)
    return points

def calc_grid_map_config(ox, oy, xy_resolution):
    """
    Calculate the grid map configuration based on obstacle coordinates.
    Args:
    ox: List of x coordinates of obstacles.
    oy: List of y coordinates of obstacles.
    xy_resolution: Resolution of the grid map.
    
    Returns:
    min_x, min_y, max_x, max_y: Grid boundaries.
    xw, yw: Grid dimensions.
    """
    min_x = round(min(ox) - EXTEND_AREA / 2.0)
    min_y = round(min(oy) - EXTEND_AREA / 2.0)
    max_x = round(max(ox) + EXTEND_AREA / 2.0)
    max_y = round(max(oy) + EXTEND_AREA / 2.0)
    xw = int(round((max_x - min_x) / xy_resolution))
    yw = int(round((max_y - min_y) / xy_resolution))
    print("The grid map is ", xw, "x", yw, ".")
    return min_x, min_y, max_x, max_y, xw, yw

def atan_zero_to_twopi(y, x):
    """
    Calculate the angle in the range [0, 2*pi].
    Args:
    y: y coordinate.
    x: x coordinate.
    
    Returns:
    angle: Angle in radians.
    """
    angle = math.atan2(y, x)
    if angle < 0.0:
        angle += math.pi * 2.0
    return angle

def init_flood_fill(center_point, obstacle_points, xy_points, min_coord, xy_resolution):
    """
    Initialize the occupancy grid using a flood fill approach.
    Args:
    center_point: Center point of the map.
    obstacle_points: Tuple of obstacle x and y coordinates.
    xy_points: Dimensions of the grid map.
    min_coord: Minimum x and y coordinates.
    xy_resolution: Resolution of the grid map.
    
    Returns:
    occupancy_map: Initialized occupancy grid map.
    """
    center_x, center_y = center_point
    prev_ix, prev_iy = center_x - 1, center_y  # Previous point for Bresenham
    ox, oy = obstacle_points
    xw, yw = xy_points
    min_x, min_y = min_coord
    occupancy_map = (np.ones((xw, yw))) * 0.5  # Initialize with unknown areas (0.5)
    for (x, y) in zip(ox, oy):
        ix = int(round((x - min_x) / xy_resolution))
        iy = int(round((y - min_y) / xy_resolution))
        free_area = bresenham((prev_ix, prev_iy), (ix, iy))  # Free area using Bresenham
        for fa in free_area:
            occupancy_map[fa[0]][fa[1]] = 0  # Mark free area (0.0)
        prev_ix = ix
        prev_iy = iy
    return occupancy_map

def flood_fill(center_point, occupancy_map):
    """
    Flood fill algorithm to identify free areas in the occupancy map.
    Args:
    center_point: Starting point for flood fill.
    occupancy_map: Occupancy grid map.
    """
    sx, sy = occupancy_map.shape
    fringe = deque()
    fringe.appendleft(center_point)  # Initialize fringe with center point
    while fringe:
        n = fringe.pop()
        nx, ny = n
        if nx > 0:  # Check left neighbor
            if occupancy_map[nx - 1, ny] == 0.5:
                occupancy_map[nx - 1, ny] = 0.0  # Mark as free area
                fringe.appendleft((nx - 1, ny))
        if nx < sx - 1:  # Check right neighbor
            if occupancy_map[nx + 1, ny] == 0.5:
                occupancy_map[nx + 1, ny] = 0.0  # Mark as free area
                fringe.appendleft((nx + 1, ny))
        if ny > 0:  # Check bottom neighbor
            if occupancy_map[nx, ny - 1] == 0.5:
                occupancy_map[nx, ny - 1] = 0.0  # Mark as free area
                fringe.appendleft((nx, ny - 1))
        if ny < sy - 1:  # Check top neighbor
            if occupancy_map[nx, ny + 1] == 0.5:
                occupancy_map[nx, ny + 1] = 0.0  # Mark as free area
                fringe.appendleft((nx, ny + 1))

def generate_ray_casting_grid_map(ox, oy, xy_resolution, breshen=True):
    """
    Generate an occupancy grid map using ray casting.
    Args:
    ox: List of x coordinates of obstacles.
    oy: List of y coordinates of obstacles.
    xy_resolution: Resolution of the grid map.
    breshen: Flag to use Bresenham's algorithm or flood fill.
    
    Returns:
    occupancy_map: Generated occupancy grid map.
    min_x, max_x, min_y, max_y: Grid boundaries.
    xy_resolution: Resolution of the grid map.
    """
    min_x, min_y, max_x, max_y, x_w, y_w = calc_grid_map_config(ox, oy, xy_resolution)
    occupancy_map = np.ones((x_w, y_w)) / 2  # Initialize with unknown areas (0.5)
    center_x = int(round(-min_x / xy_resolution))  # Map center in the grid
    center_y = int(round(-min_y / xy_resolution))  # Map center in the grid
    
    if breshen:
        # Use Bresenham's algorithm for ray casting
        for (x, y) in zip(ox, oy):
            ix = int(round((x - min_x) / xy_resolution))
            iy = int(round((y - min_y) / xy_resolution))
            laser_beams = bresenham((center_x, center_y), (ix, iy))  # Ray casting
            for laser_beam in laser_beams:
                occupancy_map[laser_beam[0]][laser_beam[1]] = 0.0  # Mark free area
            occupancy_map[ix][iy] = 1.0  # Mark obstacle area
            occupancy_map[ix + 1][iy] = 1.0  # Extend obstacle area
            occupancy_map[ix][iy + 1] = 1.0  # Extend obstacle area
            occupancy_map[ix + 1][iy + 1] = 1.0  # Extend obstacle area
    
    else:
        # Use flood fill algorithm for ray casting
        occupancy_map = init_flood_fill((center_x, center_y), (ox, oy), (x_w, y_w), (min_x, min_y), xy_resolution)
        flood_fill((center_x, center_y), occupancy_map)
        occupancy_map = np.array(occupancy_map, dtype=float)
        for (x, y) in zip(ox, oy):
            ix = int(round((x - min_x) / xy_resolution))
            iy = int(round((y - min_y) / xy_resolution))
            occupancy_map[ix][iy] = 1.0  # Mark obstacle area
            occupancy_map[ix + 1][iy] = 1.0  # Extend obstacle area
            occupancy_map[ix][iy + 1] = 1.0  # Extend obstacle area
            occupancy_map[ix + 1][iy + 1] = 1.0  # Extend obstacle area
    return occupancy_map, min_x, max_x, min_y, max_y, xy_resolution

def main():
    """
    Main function to read data, generate map and plot results.
    """
    print(__file__, "start")
    xy_resolution = 0.02  # x-y grid resolution
    ang, dist = file_read("C:/Users/mugis/Documents/Master_IN/visionBasedEmbeddesSystem/recap/lidar01.csv")
    ox = np.sin(ang) * dist  # Calculate x coordinates of obstacles
    oy = np.cos(ang) * dist  # Calculate y coordinates of obstacles
    occupancy_map, min_x, max_x, min_y, max_y, xy_resolution = generate_ray_casting_grid_map(ox, oy, xy_resolution, True)
    xy_res = np.array(occupancy_map).shape
    plt.figure(1, figsize=(10, 4))
    plt.subplot(122)
    plt.imshow(occupancy_map, cmap="PiYG_r")
    plt.clim(-0.4, 1.4)
    plt.gca().set_xticks(np.arange(-.5, xy_res[1], 1), minor=True)
    plt.gca().set_yticks(np.arange(-.5, xy_res[0], 1), minor=True)
    plt.grid(True, which="minor", color="w", linewidth=0.6, alpha=0.5)
    plt.colorbar()
    plt.subplot(121)
    plt.plot([oy, np.zeros(np.size(oy))], [ox, np.zeros(np.size(oy))], "ro-")
    plt.axis("equal")
    plt.plot(0.0, 0.0, "ob")
    plt.gca().set_aspect("equal", "box")
    bottom, top = plt.ylim()
    plt.ylim((top, bottom))
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
