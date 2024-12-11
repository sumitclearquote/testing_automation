import os
import random
import json
import numpy as np
import time

from skimage.measure import find_contours
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon as poly
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


#for car_crop
import detect_car

status_color = {'correctly-identified':(0,255,0),'false-positive':(255,0,0),'wrong-detection':(0,0,255),'non-detection':(255,255,0)}

def get_color(status,alpha):
    color = status_color[status]
    color = list(color)
    color.append(alpha)
    color = tuple(color)
    return color

def get_contour(contours):
    # print(contours)
    x1,y1,x2,y2 = contours
    cnt = [[y1,x1],[y2,x1],[y2,x2],[y1,x2],[y1,x1]]
    return cnt

def display_mask(contours, label, img, font, label_type,mask_type,status_type,shift_point=None,alpha = 255):
    # if label not in ['scratch_7']:
    #     return img
    label = label.replace('rust','')
    label = label.replace('_','')
    width, height = img.size
    temp = Image.new('RGBA', img.size, (0,0,0,0))
    canvas = ImageDraw.ImageDraw(temp)
    main_contour_list = []
    if shift_point is None:
        shift_point = (0,0)
    x_shift = shift_point[0]
    y_shift = shift_point[1]
    # print(contours)
    # print(x_shift,y_shift)
    # Check if Damage or Panel to fill masks
    background = ((23, 146, 139, alpha)) # label background - blue
    if(label_type == "damage"):
            if isinstance(contours[0],(int,float)):
                print("Contour is bounding box")
                contours = get_contour(contours)
            for i in range(len(contours)):
                main_contour_list.append(contours[i][1] + x_shift)
                main_contour_list.append(contours[i][0] + y_shift)
            # outline_color = (236, 239, 241, alpha) # scratches outline - grey
            outline_color = get_color(status_type,alpha)
            if (label != 'scratch'):
                # outline_color = (255, 200, 55, alpha) # other damages outline - yellow
                text_length = canvas.textlength(label, font=font)
                text_size = (text_length,font.size)
                canvas.rectangle([main_contour_list[0] , main_contour_list[1] - text_size[1], main_contour_list[0] + text_size[0], main_contour_list[1]], fill = background)
                canvas.text((main_contour_list[0], main_contour_list[1] - text_size[1]), label, font = font)
            canvas.line(main_contour_list, fill = outline_color, width = int(width * 0.002))
    else:
        max_verts = 0
        if(len(contours) > 1):
            for verts in contours:
                if(len(verts) > max_verts):
                    max_verts = len(verts)
                    contours = [verts]
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            verts = verts.tolist()
            for vertex in verts:
                main_contour_list.append(vertex[0])
                main_contour_list.append(vertex[1])

        fill_color = (42, 182, 182, 50) # panel background - blue
        outline_color = (42, 182, 182) # panel outline - yellow
        canvas.polygon(main_contour_list, outline = None, fill = fill_color)
        text_size = canvas.textsize(label, font = font)
        canvas.rectangle([main_contour_list[0] - 5, main_contour_list[1] - 5, main_contour_list[0] + text_size[0] + 5,
                        main_contour_list[1] + text_size[1] + 5], fill = background)
        canvas.text((main_contour_list[0], main_contour_list[1]), label, font = font)
        canvas.line(main_contour_list, fill = outline_color, width = int(width * 0.002))
    img = Image.alpha_composite(img, temp)
    return img


def draw_and_save(src_path, dest_path, label_type, threshold,test_json,data_type,car_crop):
    #src_path = os.path.join('data',direc)
    #dest_path = os.path.join('result',direc)
    os.system('mkdir -p '+os.path.join(dest_path,data_type,'gt'))
    os.system('mkdir -p '+os.path.join(dest_path,data_type,'pred'))
    path_to_result_json = os.path.join(dest_path,test_json)
    jsn = json.load(open(path_to_result_json))
    for key in tqdm(jsn.keys()):
        filename = key
        # print(filename)
        # if filename != 'o1L1J40m05_1717681305940.jpg':
        #     continue
        path_to_image = os.path.join(src_path,filename)
        gt_image = Image.open(path_to_image)
        # cropped_image, point = detect_car.crop_car(car_detector,gt_image)
        pred_image = Image.open(path_to_image)
        width, height = gt_image.size
        font = ImageFont.truetype("arial.ttf", size = int(width*0.02))
        # add 4th channel
        gtref_added  = []
        gt_image = gt_image.convert('RGBA')
        pred_image = pred_image.convert('RGBA')
        for p in jsn[key]:
            if(p['status'] == 'correctly-identified'):
                if (p['pred_bb'] != [] and p['confidence']>=threshold):
                    pred_image = display_mask(np.asarray(p['pred_mask'][0]), p['predicted_class']+str(round(p['confidence']*100,2)), pred_image, font, label_type,'pred',p['status'],p['shift-point']) #read label_type
                    gt_image = display_mask(np.asarray(p['gt_mask']), p['gtref'], gt_image, font, label_type,'gt',p['status']) #read label_type
                    gtref_added.append(p['gtref'])
                elif(p['pred_bb'] != [] and p['confidence'] <threshold) and p['gtref'] not in gtref_added:
                    gt_image = display_mask(np.asarray(p['gt_mask']), p['gtref'], gt_image, font, label_type,'gt','non-detection') # detection less than required threshold are non-detection.
                    gtref_added.append(p['gtref'])
            elif(p['status'] == 'wrong-detection'): #actual and pred
                if p['gtref'] not in gtref_added:
                    gt_image = display_mask(np.asarray(p['gt_mask']), p['gtref'], gt_image, font, label_type,'gt',p['status'])
                    gtref_added.append(p['gtref'])
                if(p['pred_bb']!=[] and p['confidence']>threshold):
                    pred_image = display_mask(np.asarray(p['pred_mask'][0]), p['predicted_class']+str(round(p['confidence']*100,2)), pred_image, font, label_type,'pred',p['status'],p['shift-point'])

            elif(p['status'] == 'non-detection' ):
                gt_image = display_mask(np.asarray(p['gt_mask']), p['gtref'], gt_image, font, label_type,'gt',p['status'])

            elif(p['status'] == 'false-positive'):
                if(p['pred_bb']!=[] and p['confidence']>threshold):
                    pred_image = display_mask(np.asarray(p['pred_mask'][0]), p['predicted_class']+str(round(p['confidence']*100,2)), pred_image, font, label_type,'pred',p['status'],p['shift-point'])

        gt_image = gt_image.convert('RGB')
        pred_image = pred_image.convert('RGB')
        path_to_gt_image = os.path.join(dest_path,data_type,'gt',filename)
        path_to_pred_image = os.path.join(dest_path,data_type,'pred',filename)
        gt_image.save(path_to_gt_image)
        pred_image.save(path_to_pred_image)
        # print('Fin...',filename,flush=True)

if __name__ == '__main__':
    # store_name_list = ['detectron_prodmsil']
    # dataset_list=['MSIL_10_config']

    #store_name_list = ['model_8.1K_dirt_test']
    store_name_list = ['model_exp3_1999_dirt_test']
    dataset_list=['dirt_test']

    to_save = {j:i for i,j in zip(dataset_list,store_name_list)}
    dest = '../testing/dirt_exp3/'
    mask_threshold = 0.4
    car_crop = True
    for test_json in os.listdir(dest):
        # print(test_json)
        src = '../test_data/'
        if test_json.endswith('json'):
            data_type = os.path.splitext(test_json)[0]
            if data_type not in store_name_list:
                print(data_type)
                continue
            src = os.path.join(src,to_save[data_type])
            draw_and_save(src, dest, 'damage',mask_threshold,test_json,data_type,car_crop)
