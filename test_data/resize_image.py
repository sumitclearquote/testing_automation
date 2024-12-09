import cv2
import os
import imutils
import json
import copy
from tqdm import tqdm
src = 'dirt_test'


resize_width_list = [1920]
dest_list = [src+'_resized_'+str(i) for i in resize_width_list]
print(dest_list)

for resize_width,dest in zip(resize_width_list,dest_list):
    print(resize_width,dest)
    os.makedirs(dest,exist_ok=True)
    new_jsn = {}
    with open(os.path.join(src,'via_region_data.json')) as f:
        jsn  =json.load(f)
    
    for i , key in tqdm(enumerate(list(jsn.keys())),total=len(jsn.keys())):
        new_jsn[key] = copy.deepcopy(jsn[key])
        filename = jsn[key]['filename']
        filename = filename.replace("https://cq-workflow.s3.ap-south-1.amazonaws.com/", "")
        filename = filename.replace('https://cq-workflow.s3.amazonaws.com/','')
        filename = os.path.basename(filename)

        ## resize the image.
        test_image = cv2.imread(os.path.join(src,filename))
        try:
            h,w = test_image.shape[:2]
        except:
            print("File not found")
            print(os.path.join(src,filename))
            continue
        new_image = imutils.resize(test_image,width=resize_width)
        cv2.imwrite(os.path.join(dest,filename),new_image)
        ratio = float(new_image.shape[1])/test_image.shape[1]
        # print(ratio)

        new_regions = []
        for i,region in enumerate(new_jsn[key]['regions']):
            # print(region)
            try:
                all_points_x = region['shape_attributes']['all_points_x']
                all_points_y = region['shape_attributes']['all_points_y']
            except:
                continue
            all_points_x = [int(i*ratio) for i in all_points_x]
            # print(all_points_y)
            all_points_y = [int(i*ratio) for i in all_points_y]
            # print(all_points_y)
            region['shape_attributes']['all_points_x'] = all_points_x
            region['shape_attributes']['all_points_y'] = all_points_y
            # print(region)
            new_regions.append(region)
            # assert False   
        new_jsn[key]['regions'] = new_regions
    
    with open(os.path.join(dest,'via_region_data.json'),'w') as f:
        json.dump(new_jsn,f)
    # break


