
import masking_utils
import result_csv
import time
import os
import cv2
import pickle
import json
import copy
from json import JSONEncoder
import random
import numpy as np
from PIL import Image
from PIL import ImageFile


from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.structures import BoxMode, instances
from detectron2.projects import point_rend
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor

#___________________________________________________________________
import datetime
import numpy as np
import skimage.draw
import random
import time
import urllib.request
import urllib.parse
from prettytable import PrettyTable
from os.path import splitext, basename
ImageFile.LOAD_TRUNCATED_IMAGES = True
#_____________________________________________________________________
from skimage.measure import find_contours
from shapely.geometry import Point, box
from shapely.geometry import Polygon as poly
from shapely.affinity import translate
from PIL import Image, ImageDraw, ImageFont
# import mmcv
# from mmdet.apis import init_detector, inference_detector

import mask_overlap
import get_class
import detect_car
#import detector
import torch

#import modelLoaderClasses
###############################################################################################################3
# import required functions, classes

RUN_ON_CPU = False

if not RUN_ON_CPU:
    GPU_PRESENT = torch.cuda.is_available()
else:
    GPU_PRESENT = False

model_path = "../../detectron2/mahindra_dirt/exp3/model_0003999.pth"

def load_dirt():
    cfg = get_cfg()
    cfg.merge_from_file("../../detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml") 
    #cfg.MODEL.WEIGHTS = '../model_weights/exp1_model_10.8K.pth'
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8, 16, 32, 64, 128, 256, 512]]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 1.33, 1.5, 2.0]]
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 1024
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1
    cfg.MODEL.DEVICE = 'cpu' if RUN_ON_CPU else 'cuda:0'
    predictor = DefaultPredictor(cfg)
    return predictor
    

def load_car_detector():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = os.path.join('../model_weights/car_crop_V_2.pth')
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[16, 32, 64, 128, 256, ]]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 0.75, 1.0, 1.5, 2.0]]
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 1024
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.INPUT.MIN_SIZE_TEST = 512
    cfg.INPUT.MAX_SIZE_TEST = 800
    cfg.MODEL.DEVICE = 'cpu' if RUN_ON_CPU else 'cuda:0'
    predictor = DefaultPredictor(cfg)
    return predictor

dirt_predictor = load_dirt()
#car_detector = load_car_detector()

def detectron_prediction(predictor,test_image):
    
    combined_outputs = predictor(test_image)
    combined_out = combined_outputs["instances"].to("cpu")
    combined_all_fields = combined_out.get_fields()
    p = {}
    p['class_ids'] = combined_all_fields['pred_classes'].numpy()
    p['masks'] = combined_all_fields['pred_masks'].numpy()
    p['rois'] = combined_all_fields['pred_boxes'].tensor.numpy()
    p['scores'] = combined_all_fields['scores'].numpy()
    torch.cuda.empty_cache()
    return p


model_classes = ['dirt']

print('All models loaded')
# exit()

def compare_damages_generic_detectron2(data, dest, iou_threshold, damage_list,gt_class_list,
                                       save_name, remove_overlapping_fps, remove_duplicate_fps, overlap_mode='bbox'):

    d = {}
    # manual_sheet = {}
    manual_sheet_list= []
    limit_dict = {'panel':0.1,'others':0.1}
    print(limit_dict)
    # manual_column_list = ['GT']
    manual_sheet_list.append('')
    os.system('mkdir -p '+dest)
    path_to_json = os.path.join(data,'via_region_data.json')
    # path_to_json = os.path.join(data,'via_region_data.json')
    jsn = json.load(open(path_to_json))
    cnt = 0
    print("NO OF IMAGES",len(jsn))
    for iii,key in enumerate(list(jsn.keys())):

        filename = jsn[key]['filename']
        filename = filename.replace("https://cq-workflow.s3.ap-south-1.amazonaws.com/", "")
        filename = filename.replace('https://cq-workflow.s3.amazonaws.com/','')
        filename = os.path.basename(filename)

        
        # if filename != '187496_Damage%20View_9_unmarkedImage_Images_1520335986933_1566890109162.jpeg': continue
        path_to_image = os.path.join(data,filename)
        test_image = cv2.imread(path_to_image)

        try:
            height, width, channels = test_image.shape
        except:
            continue
        #image for classifier
        input_img = Image.open(path_to_image)
        input_img = input_img.convert('RGB')
        
        ratio = 1 
        point = None
        #------------------------Cropping logic--------------------------------------------------
        # cropped_image, point = detect_car.crop_car(car_detector,test_image)
        # if cropped_image is not None:
        #     test_image = cropped_image
        # print("Cropping successful",point)
        #----------------------------------------------------------------------------------------
        d[filename] = []
        print(iii, ' Processing:',path_to_image,flush=True)

       
        panels_dict = detectron_prediction(dirt_predictor,test_image)
        final_rois,final_scores,final_class_ids,mid_masks,predicted_class_names = masking_utils.process_damage_dict(panels_dict,damage_list,limit_dict)


        #remove duplicate FPs when FPs are overlapping with each other.
        if remove_overlapping_fps:
            final_rois,final_scores,final_class_ids,mid_masks,predicted_class_names= masking_utils.remove_duplicate_damages(final_rois,final_class_ids, final_scores,mid_masks,predicted_class_names) 

        if not len(mid_masks):
            mid_masks = [None for i in range(len(final_rois))]

        predicted_status_list = [False for i in range(len(predicted_class_names))] #completion list
        
        
        ###### find list of only damages
        region = jsn[key]['regions']
        gt_list = []
        for k in range(len(region)):
            try:
                region_name  = region[k]['region_attributes']['identity']
                region_name = region_name.lower()
                if region_name.endswith(' '):
                    region_name = region_name.replace(' ','')
                region[k]['region_attributes']['identity'] = region_name
            except:
                continue
            if region_name in gt_class_list:
                gt_list.append(region[k])
        print("Damages in GT: ",len(gt_list))
        # filter gt here according to panels or views:
        gt_status_list = [False for i in range(len(gt_list))]
        # print([i['region_attributes']['identity'] for i in gt_list])
        # print(predicted_class_names)
        for i in range(len(predicted_class_names)):
            #  traverse every GT region
            # print('Predicted',predicted_class_names[i])

            for reg in range(len(gt_list)):
                actual_class = gt_list[reg]['region_attributes']['identity']


                x_list = gt_list[reg]['shape_attributes']['all_points_x']
                x_list = [int(i*ratio) for i in x_list]
                y_list = gt_list[reg]['shape_attributes']['all_points_y']
                y_list = [int(i*ratio) for i in y_list]

                if overlap_mode == 'mask':
                    coord = zip(x_list,y_list)
                    p1 = poly(coord)
                else:
                    p1 = box(min(x_list),min(y_list),max(x_list),max(y_list))
                try:
                    if overlap_mode == 'mask':
                        contour  =  mask_contour(mid_masks[i])
                        if (contour == []):
                            continue
                        p2 = poly(contour[0][:,::-1]) #reverse y,x -> x,y
                    else:
                        bbox = final_rois[i]
                        p2 = box(bbox[0],bbox[1],bbox[2],bbox[3])  #x1 y1 x2 y2
                    if point is not None:
                        # print("Shift point ", point)
                        p2 = translate(p2,xoff=point[0],yoff=point[1]) # shifting the point to compensate for cropping.
                except Exception as e:
                    print('P1:',p1)
                    print('P2:',p2)
                    print('i',i,'reg',reg)
                    print(e)
                iou = mask_overlap.IntersectionOverUnion(p1,p2)

                # # gt bounding box
                x1,y1,x2,y2 = min(x_list),min(y_list),max(x_list),max(y_list)
                gt_bb = [x1,y1,x2,y2]
                gt_mask = [[y,x] for x,y in zip(x_list,y_list)]

                actual_class = transform_actual_class(actual_class)
                # print("After transformation", actual_class)

                numbered_actual_class = actual_class+'_'+str(reg)	
                # print(predicted_class_names[i], actual_class,numbered_actual_class,iou)

                if (iou >= iou_threshold):
                    if(predicted_class_names[i] == 'd2' and actual_class in ['d1','d2','d3','bumperdent']):
                        d[filename].append({'gtref':numbered_actual_class,'actual_class':actual_class,'predicted_class':predicted_class_names[i],
                        'status':'correctly-identified', 'IoU':iou, 'confidence':final_scores[i],
                        'gt_bb':gt_bb,'pred_bb':final_rois[i],'gt_mask':gt_mask,'pred_mask':mask_contour(mid_masks[i]),"shift-point":point})
                        gt_status_list[reg] = True
                        predicted_status_list[i] = True
                        continue

                    elif(predicted_class_names[i] == 'tear' and actual_class in ['tear','bumpertear','bumpertorn']):
                        d[filename].append({'gtref':numbered_actual_class,'actual_class':actual_class,'predicted_class':predicted_class_names[i],
                        'status':'correctly-identified', 'IoU':iou, 'confidence':final_scores[i],
                        'gt_bb':gt_bb,'pred_bb':final_rois[i],'gt_mask':gt_mask,'pred_mask':mask_contour(mid_masks[i]),"shift-point":point})
                        gt_status_list[reg] = True
                        predicted_status_list[i] = True
                        continue
                    
                    elif(predicted_class_names[i] == 'broken' and actual_class in ['broken','cracked']):
                        d[filename].append({'gtref':numbered_actual_class,'actual_class':actual_class,'predicted_class':predicted_class_names[i],
                        'status':'correctly-identified', 'IoU':iou, 'confidence':final_scores[i],
                        'gt_bb':gt_bb,'pred_bb':final_rois[i],'gt_mask':gt_mask,'pred_mask':mask_contour(mid_masks[i]),"shift-point":point})
                        gt_status_list[reg] = True
                        predicted_status_list[i] = True
                        continue

                    # clipsbroken or shattered
                    elif(predicted_class_names[i] == actual_class):
                        d[filename].append({'gtref':numbered_actual_class,'actual_class':actual_class,'predicted_class':predicted_class_names[i],
                        'status':'correctly-identified', 'IoU':iou, 'confidence':final_scores[i],
                        'gt_bb':gt_bb,'pred_bb':final_rois[i],'gt_mask':gt_mask,'pred_mask':mask_contour(mid_masks[i]),"shift-point":point})
                        gt_status_list[reg] = True
                        predicted_status_list[i] = True
                        continue
        # wrong-detections______________________________________________________________________________________________________________________
        # print("Now processing wrong detections ")
        # print(predicted_status_list)
        # print(gt_status_list)
        for i in range(len(predicted_status_list)):
            if(not(predicted_status_list[i])):
                for reg in range(len(gt_status_list)):
                    x_list = gt_list[reg]['shape_attributes']['all_points_x']
                    x_list =  [int(i*ratio) for i in x_list]
                    y_list = gt_list[reg]['shape_attributes']['all_points_y']
                    y_list = [int(i*ratio) for i in y_list]
                    x1,y1,x2,y2 = min(x_list),min(y_list),max(x_list),max(y_list)
                    gt_bb = [x1,y1,x2,y2]
                    gt_mask = [[y,x] for x,y in zip(x_list,y_list)]
                    

                    actual_class = gt_list[reg]['region_attributes']['identity']
                    actual_class = transform_actual_class(actual_class)
                    
                    if overlap_mode == 'mask':
                        coord = zip(x_list,y_list)
                        p1 = poly(coord)
                    else:
                        p1 = box(min(x_list),min(y_list),max(x_list),max(y_list))
                    try:
                        if overlap_mode == 'mask':
                            contour  =  mask_contour(mid_masks[i])
                            if (contour == []):
                                continue
                            p2 = poly(contour[0][:,::-1]) #reverse y,x -> x,y
                        else:
                            bbox = final_rois[i]
                            p2 = box(bbox[0],bbox[1],bbox[2],bbox[3])  #x1 y1 x2 y2
                        if point is not None:
                            # print("Shift point", point)
                            p2 = translate(p2,xoff=point[0],yoff=point[1])
                    except Exception as e:
                        print('P1:',p1)
                        print('P2:',p2)
                        print('i',i,'reg',reg)
                        print(e)
                    iou = mask_overlap.IntersectionOverUnion(p1,p2)
                    # print(iou)
                    # print(actual_class,predicted_class_names[i])
                    numbered_actual_class = actual_class+'_'+str(reg)
                    # overlap > 0.5 but class mismatch

                    if( iou>=iou_threshold and (predicted_class_names[i] != actual_class)):
                        d[filename].append({'gtref':numbered_actual_class,'actual_class':actual_class,'predicted_class':predicted_class_names[i],
                        'status':'wrong-detection', 'IoU':iou, 'confidence':final_scores[i],
                        'gt_bb':gt_bb,'pred_bb':final_rois[i],'gt_mask':gt_mask,'pred_mask':mask_contour(mid_masks[i]),"shift-point":point})
                        gt_status_list[reg] = True
                        predicted_status_list[i] = True
    # false-positive_________________________________________________________________________________________________________________________
        for j in range(len(predicted_status_list)):
            if not(predicted_status_list[j]): #pred box only
                d[filename].append({'gtref':'','actual_class':'','predicted_class':predicted_class_names[j],
                'status':'false-positive','IoU':'','confidence':final_scores[j],
                'gt_bb':'','pred_bb':final_rois[j],'gt_mask':'','pred_mask':mask_contour(mid_masks[j]),"shift-point":point}) #'IoU':iou, 'confidence':final_scores[i]})
                predicted_status_list[j] = True
    # non-detection___________________________________________________________________________________________________________________________
        for j in range(len(gt_status_list)):
            if not(gt_status_list[j]):
                x_list = gt_list[j]['shape_attributes']['all_points_x']
                x_list = [int(i*ratio) for i in x_list]
                y_list = gt_list[j]['shape_attributes']['all_points_y']
                y_list = [int(i*ratio) for i in y_list]
                #gt bounding box only
                x1,y1,x2,y2 = min(x_list),min(y_list),max(x_list),max(y_list)
                gt_bb = [x1,y1,x2,y2]
                gt_mask = [[y,x] for x,y in zip(x_list,y_list)]
                actual_class = gt_list[j]['region_attributes']['identity']
                actual_class = transform_actual_class(actual_class)
                numbered_actual_class = actual_class+'_'+str(j)
                d[filename].append({'gtref':numbered_actual_class,'actual_class':actual_class,'predicted_class':'',
                'status':'non-detection','IoU':'','confidence':'',
                'gt_bb':gt_bb,'pred_bb':'','gt_mask':gt_mask,'pred_mask':'',"shift-point":point}) #'IoU':iou'confidence':final_scores[i]
                gt_status_list[j] = True

        if remove_duplicate_fps: 
            print("Removing false positives for image: ",filename)
            d[filename] = masking_utils.remove_wrong_false_positive(d[filename])
        # break
# store json________________________________________________________________________________________________________________________________________________
    # exit()
    path_to_res_json = os.path.join(dest,save_name+'.json')
    with open(path_to_res_json,'w')  as f:
        f.write(json.dumps(d,indent=5,cls=NumpyArrayEncoder))
        f.close()
    # with open(os.path.join(dest,'new_damage_detected.json'),'w') as f:
    # 	json.dump(damage_detected,f,cls=NumpyArrayEncoder)
    return d

def transform_actual_class(actual_class,divided=False): # divided is used if we are generating testing automation results using size.
    # print(f"Arguments Recieved: {actual_class} and {divided}")
    if not divided:
        if(actual_class in ['d1','d2','d3','bumperdent']):
            return 'd2'
        elif(actual_class in ['tear','bumpertear','bumpertorn']):
            return 'tear'
        elif(actual_class in ['broken','cracked']):
            return 'broken'
        elif(actual_class in ['rust','paintedrust']):
            return 'rust'
        elif actual_class in ['dirt', 'bird_dropping']:
            return 'dirt'
        else:
            return actual_class
    else:
        if any(i in actual_class for i in ['S','M','L']):
            base_damage = actual_class.split('_')[0]
            st = actual_class.split('_')[-1]
            return transform_actual_class(base_damage,False) + "_"+st
        else:
            return transform_actual_class(actual_class,False)

from json import JSONEncoder
class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyArrayEncoder, self).default(obj)



def mask_contour(mask):
        if mask is None:
            return mask
        padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        # contours = find_contours(padded_mask, 0.9)
        contours = find_contours(mask,0.9)
        return contours

if __name__ == '__main__':
    make = 'any'
    model = 'any'
    weights_type = 'detectron2'

    #HPs:
    remove_overlapping_fps = True #remove duplicate FPs when FPs are overlapping with each other.
    remove_duplicate_fps = True #Remove duplicate FPs when More than one detections are overlapping over GT but IOU of one detection is < threshold with the GT but higher with the other detection.
    
    iou_threshold = 0.1
    store_name_list = ['dirt_test'] # The suffix to add to MODEL_NAME variable.
    dataset_list=['dirt_test'] # Add if more than one dataset in "test_data" folder
    to_save = {i:j for i,j in zip(dataset_list,store_name_list)}

    for dataset in dataset_list:
        data = '../test_data'
        data = os.path.join(data,dataset)
        dest = '../testing/dirt_exp3/' #
        #model_name  = 'model_exp1_10.8K_'
        model_name = "model_exp3_3999_"
        gt_class_list = ['dirt', 'bird_dropping'] # add dirt, bird_dropping
        pred_list = ['dirt']
        save_name = model_name + to_save[dataset] 
        path_to_csv = os.path.join(dest,save_name+'.csv')
        print("Path to csv: ",path_to_csv)
        print("Dataset-used: ",data)
        print("dest: ",dest)
        # continue
        d = compare_damages_generic_detectron2(data, dest, iou_threshold,pred_list,gt_class_list,
                                               save_name, remove_overlapping_fps, remove_duplicate_fps)

        result_csv.write_to_csv(d,path_to_csv)
        # break
