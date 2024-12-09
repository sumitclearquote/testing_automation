
from shapely.geometry import Polygon as poly
import numpy as np
# def reduce_points(coordinates):
#     	##convert to int and flip the coordinates from (y,x) to (x,y)
# 	coordinates = np.array(coordinates,np.int32)
# 	coordinates = np.fliplr(coordinates)
# 	reduce_factor = int(len(coordinates)/50)
# 	reduce_factor = max(reduce_factor,1)
# 	new_coords = []
# 	for i in range(len(coordinates)):
# 		if i%reduce_factor == 0 :
# 			new_coords.append(list(coordinates[i]))
# 	return new_coords

def coords(p2):
    x,y =  p2.exterior.coords.xy
    reduce_factor = int(len(x)/50)
    reduce_factor = max(reduce_factor,1)
    new_coords = []
    
    for i in range(len(x)):
        if i % reduce_factor == 0 :
            new_coords.append([x[i], y[i]])
    
    return new_coords

# p1 is original mask from annotations
# p2 is detected mask from model
# p2 reduce the points of p2 and then make polygon again and then calculate IOU
def IntersectionOverUnion(p1,p2):
    if not(p1.is_valid):
        p1 = p1.convex_hull
    
    if not(p2.is_valid):
        p2 = p2.convex_hull
    
    if p1.intersects(p2):
        #p2 = poly(coords(p2))
        
        # #Edge Smotthing of masks
        # print("len P2: ", len(p2.exterior.coords))
        # p2 = p2.simplify(0.2, preserve_topology=False)
        # print("len P2 New: ", len(p2.exterior.coords))
        
        intersection = p1.intersection(p2).area
        union = p1.union(p2).area
        iou = intersection/union
    
    else:
        iou = 0
    return iou