import numpy as np
import torch
import masking_utils
import cv2
from prettytable import PrettyTable
# try:
# import mmcv
# from mmdet.apis import inference_detector
from sahi.predict import get_sliced_prediction, get_prediction

import copy

SKIP_DAMAGE_PANELS = ['frontws', 'leftbootlamp', 'leftheadlamp', 'lefttaillamp',
					  'rearws', 'rightbootlamp' 'rightheadlamp', 'righttaillamp',
					  'footstep', 'alloywheel', 'tyre', 'leftfrontdoorglass',
					  'leftfrontventglass', 'leftquarterglass', 'leftreardoorglass',
					  'leftrearventglass', 'licenseplate', 'rightfrontdoorglass',
					  'rightfrontventglass', 'rightquarterglass',  'rightreardoorglass',
					  'rightrearventglass', 'wheelcap', 'wheelrim', 'sunroof', 'wiper', 'indicator']

def get_damage_predictions(predictor,img_path,threshold):
	if isinstance(img_path,str):
		img = mmcv.imread(img_path)
	else: # if path is already numpy image.
		img = img_path
	h,w = img.shape[:2]
	result = inference_detector(predictor,img)
	if isinstance(result, tuple):
		bbox_res, segm_res = result
		if isinstance(segm_res, tuple):
			segm_res = segm_res[0]  # ms rcnn
	else:
		bbox_res, segm_res = result, None
	bboxes = np.vstack(bbox_res)
	labels = [np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in enumerate(bbox_res)]
	labels = np.concatenate(labels)
	segms = None
	if segm_res is not None and len(labels) > 0:
		segms = mmcv.concat_list(segm_res)
		segms = np.stack(segms,axis=0)
	scores = bboxes[:,-1]
	filter_detection = scores >= threshold
	bboxes = bboxes[filter_detection, :]
	bboxes = bboxes[:,:-1]
	labels = labels[filter_detection]
	scores = scores[filter_detection]
	if segms is not None:
		segms = segms[filter_detection, ...]
	pred_dict = {}
	pred_dict['scores']= scores
	pred_dict['class_ids']= labels
	pred_dict['rois'] = bboxes
	if segms is not None:
		pred_dict['masks'] = segms
	if 'masks' not in pred_dict and len(scores) == 0:
		pred_dict['masks'] = np.empty((0,h,w))
	return pred_dict

def predict(test_image,predictor,transformer=False):
	# try:
	if not transformer:
		panel_outputs = predictor(test_image)
		# except Exception as e:
		# 	print("Panel prediction failed with Error: ", str(e), flush=True)
		panels_dict = {}
		out = panel_outputs["instances"].to("cpu")
		all_fields = out.get_fields()
		panels_dict['class_ids'] = all_fields['pred_classes'].numpy()
		if 'pred_masks' in all_fields:
			panels_dict['masks'] = all_fields['pred_masks'].numpy()
		panels_dict['rois'] = all_fields['pred_boxes'].tensor.numpy()
		panels_dict['scores'] = all_fields['scores'].numpy()
	else:
		panels_dict = get_damage_predictions(predictor, test_image, 0.1)
	return panels_dict


def predictSahiDamage(sahi_transformer_damage_predictor, cropped_image):
    h, w = cropped_image.shape[:2]
    big = 1200
    result = None

    print("Input SAHI Image Size, H, W: ", h, " ", w, flush=True)
    cropped_image_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)  #RGB changing need to put same in PROD
    print("Image converted to RGB amd Tile size: ", big, flush=True)
    result = get_sliced_prediction(
                                cropped_image_rgb,
                                sahi_transformer_damage_predictor,
                                slice_height = big,
                                slice_width = big,
                                overlap_height_ratio = 0.2,
                                overlap_width_ratio = 0.2)

    print("Damages Detected by SAHI: ", len(result.object_prediction_list), flush=True)  
    torch.cuda.empty_cache()

    #putting results back to Transformer format
    damage_dict= {}
    bboxes = []
    scores = []
    labels = []
    segms = []
    
    for each_out in result.object_prediction_list:
        bboxes.append(each_out.bbox.to_voc_bbox())
        scores.append(each_out.score.value)
        labels.append(each_out.category.id)
        segms.append(each_out.mask.bool_mask)
            
    damage_dict['scores'] = np.array(scores)
    damage_dict['class_ids'] = np.array(labels)
    damage_dict['rois'] = np.array(bboxes)

    if len(scores) == 0:
        damage_dict['masks'] = np.empty((0, h, w))  
    else:
        damage_dict['masks'] = np.array(segms)
    return damage_dict



def predict_damage(generic_predictor,van_predictor,sahi_predictor,test_image,full_image,flip,car_crop,car_crop_flip,panel_crop,panel_crop_flip,run_van_model,panels_dict,panels,use_transformer=True,sahi_inference=False):
	print('INSIDE DAMAGE PREDICTION')
	print(f"Using transformer {use_transformer}")
	panel_wise_damage_predictor = generic_predictor
	final_damage_dict = None # default value as None
	if sahi_inference:
		flip = False
		run_van_model = False
		panel_crop = False
		panel_crop_flip = False
	
	if run_van_model:
		flip = False
	height, width, channels = test_image.shape
	image_template = np.full((height,width),False)
	if full_image:
		p = predict(test_image,generic_predictor,use_transformer)
		# print("Normal Image Inference Info")
		# masking_utils.print_shape(p)
		if final_damage_dict is None:
			final_damage_dict = copy.deepcopy(p)
		else:
			if len(p['scores']): #only merging when there are some detections.
				for key in p:
					final_damage_dict[key] = np.concatenate((final_damage_dict[key],p[key]),axis=0)
		# print("Final Info")
		# masking_utils.print_shape(final_damage_dict)
		print("Original_Detections: ",len(final_damage_dict['rois']),flush=True)
	
	if flip:
		test_flip = cv2.flip(test_image,1)
		p = predict(test_flip,generic_predictor,use_transformer)
		# print("Full Image Flip Info")
		p = masking_utils.flip_rois(p)
		if final_damage_dict is None:
			final_damage_dict = copy.deepcopy(p)
		else:
			if len(p['scores']): #only merge when there are some detections.
				final_damage_dict['class_ids'] = np.concatenate((final_damage_dict['class_ids'],p['class_ids']),axis=0)
				final_damage_dict['scores'] = np.concatenate((final_damage_dict['scores'],p['scores']),axis=0)
				final_damage_dict['rois'] = np.concatenate((final_damage_dict['rois'],p['rois']),axis=0)
				masks = []
				for i in range(len(p['scores'])):
					masks.append(np.fliplr(p['masks'][i,:,:]))
				final_damage_dict['masks'] = np.concatenate((final_damage_dict['masks'],np.array(masks)),axis=0)
		print("After_Flipping: ",len(final_damage_dict['rois']),flush=True)

	if run_van_model:
		p = predict(test_image,van_predictor,use_transformer)
		if final_damage_dict is None:
			final_damage_dict = copy.deepcopy(p)
		else:
			if len(p['scores']): #only merge when there are some detections.
				final_damage_dict['class_ids'] = np.concatenate((final_damage_dict['class_ids'],p['class_ids']),axis=0)
				final_damage_dict['scores'] = np.concatenate((final_damage_dict['scores'],p['scores']),axis=0)
				final_damage_dict['rois'] = np.concatenate((final_damage_dict['rois'],p['rois']),axis=0)
				final_damage_dict['masks'] = np.concatenate((final_damage_dict['masks'],p['masks']),axis=0)
		print("After Van Damage Model: ",len(final_damage_dict['rois']),flush=True)
	
	if panel_crop or panel_crop_flip:
		
		if panel_crop and (panels_dict != {} or panels_dict is not None):
			print("Inside Panel Crop",flush=True)
			# go through all panels detected, if any have size greater than 1000*1000 then do one more detection on it.
			for i in range(len(panels_dict['scores'])):
				x1,y1, x2,y2 =  panels_dict['rois'][i] # get the bounding box of panel.
				x1,y1, x2,y2 = int(round(x1)), int(round(y1)),int(round(x2)),int(round(y2))
				if x1 < 0: x1 = 0
				if x2  > width: x2 = width - 1
				print("Panel is :",panels[panels_dict['class_ids'][i]], "Dimensions: ",x2-x1,'x',y2-y1,flush=True)
				if (x2-x1) >= 800 and (y2-y1) >= 800 and panels[panels_dict['class_ids'][i]] not in SKIP_DAMAGE_PANELS:
					print("Cropping panel:",panels[panels_dict['class_ids'][i]],flush=True)
					cropped_panel = test_image[y1:y2,x1:x2]
					shift_point = (x1,y1)
					#do  damage inference on this cropped panel.
					try:
						panel_crop_damages = panel_wise_damage_predictor(cropped_panel)
					except Exception as e:
						print("Error in panel_crop damage detection ", str(e),flush=True)
					panel_out = panel_crop_damages['instances'].to("cpu")
					all_fields = panel_out.get_fields()
					panel_crop = {}
					panel_crop['class_ids'] = all_fields['pred_classes'].numpy()
					panel_crop['masks'] = all_fields['pred_masks'].numpy()
					panel_crop['rois'] = all_fields['pred_boxes'].tensor.numpy()
					panel_crop['scores'] = all_fields['scores'].numpy()
					# adjust cropped masks to original image.
					if len(panel_crop['scores']): #if there are any detections
						detected_masks = list(panel_crop['masks'])
						panel_crop['masks'] = masking_utils.masks_adjustment(detected_masks,image_template,shift_point)
					# print("Panel Crop Info")
					# masking_utils.print_shape(panel_crop)
					print(f"Damages detected in panel {panels[panels_dict['class_ids'][i]]} are {len(panel_crop['scores'])}")
					if final_damage_dict is None or len(final_damage_dict['rois'])==0:
						final_damage_dict = copy.deepcopy(panel_crop)
					else: 
						if len(panel_crop['scores']):
							for key in panel_crop:
								final_damage_dict[key] = np.concatenate((final_damage_dict[key],panel_crop[key]),axis=0)
					# print("Final Info")
					# masking_utils.print_shape(final_damage_dict)
					print("After Panel Cropping: ", len(final_damage_dict['rois']),flush=True)
		
		if panel_crop_flip:
			print("Inside panel_crop flip",flush=True)
			for i in range(len(panels_dict['scores'])):
				x1,y1, x2,y2 =  panels_dict['rois'][i] # get the bounding box of panel.
				x1,y1, x2,y2 = int(round(x1)), int(round(y1)),int(round(x2)),int(round(y2))
				if x1 < 0: x1 = 0
				if x2  > width: x2 = width - 1
				if (x2-x1) >= 800 and (y2-y1) >= 800 and panels[panels_dict['class_ids'][i]] not in SKIP_DAMAGE_PANELS:
					cropped_panel = test_image[y1:y2,x1:x2]
					cropped_panel = cv2.flip(cropped_panel,1) # flipping performed on panel_cropped.
					shift_point = (x1,y1)
					#do  damage inference on this cropped panel.
					try:
						panel_crop_damages = panel_wise_damage_predictor(cropped_panel)
					except Exception as e:
						print("Error in panel_crop damage detection ", str(e),flush=True)
					panel_out = panel_crop_damages['instances'].to("cpu")
					all_fields = panel_out.get_fields()
					panel_crop = {}
					panel_crop['class_ids'] = all_fields['pred_classes'].numpy()
					panel_crop['masks'] = all_fields['pred_masks'].numpy()
					panel_crop['rois'] = all_fields['pred_boxes'].tensor.numpy()
					panel_crop['scores'] = all_fields['scores'].numpy()
					panel_crop = masking_utils.flip_rois(panel_crop)
					d_masks = []
					for i in range(len(panel_crop['rois'])):
						d_masks.append(np.fliplr(panel_crop['masks'][i,:,:]))
					# adjust cropped masks to original image.
					if len(panel_crop['scores']): #if there are any detections
						detected_masks = copy.deepcopy(d_masks)
						panel_crop['masks'] = masking_utils.masks_adjustment(detected_masks,image_template,shift_point)
					print("Panel Crop  Flip Info")
					masking_utils.masking_utils.print_shape(panel_crop)
					if final_damage_dict is None:
						final_damage_dict = copy.deepcopy(panel_crop)
					else:
						if len(panel_crop['scores']):
							for key in panel_crop:
								final_damage_dict[key] = np.concatenate((final_damage_dict[key],panel_crop[key]),axis=0)
					print("Final Info")
					masking_utils.print_shape(final_damage_dict)
		print("After  Panel Cropping and Flipping: ", len(final_damage_dict['rois']),flush=True)		
	
		if final_damage_dict is None:
			final_damage_dict['rois'] = []
			final_damage_dict['scores'] = []
			final_damage_dict['masks'] = []
			final_damage_dicts['class_ids']= []
	
	
	if sahi_inference:
		p = predictSahiDamage(sahi_predictor,test_image)
		if final_damage_dict is None:
			final_damage_dict = copy.deepcopy(p)
		else:
			if len(p['scores']): #only merge when there are some detections.
				final_damage_dict['class_ids'] = np.concatenate((final_damage_dict['class_ids'],p['class_ids']),axis=0)
				final_damage_dict['scores'] = np.concatenate((final_damage_dict['scores'],p['scores']),axis=0)
				final_damage_dict['rois'] = np.concatenate((final_damage_dict['rois'],p['rois']),axis=0)
				final_damage_dict['masks'] = np.concatenate((final_damage_dict['masks'],p['masks']),axis=0)
		print("After Sahi Model: ",len(final_damage_dict['rois']),flush=True)
	
	damage_list=['scratch', 'd2', 'tear', 'clipsbroken',  'shattered', 'broken']
	ttt = PrettyTable(["Class_Name","Confidence"])
	for i in range(len(final_damage_dict['scores'])):
		ttt.add_row([damage_list[final_damage_dict['class_ids'][i]],final_damage_dict['scores'][i]])
	print(ttt,flush=True)
	return final_damage_dict