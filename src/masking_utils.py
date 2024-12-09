import numpy as np
import mask_overlap
from skimage.measure import find_contours
from shapely.affinity import translate
from shapely.geometry import Polygon
from prettytable import PrettyTable
import copy
import json


def mask_contour(mask):
	padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
	padded_mask[1:-1, 1:-1] = mask
	# contours = find_contours(padded_mask, 0.9)
	contours = find_contours(mask,0.9)

	return contours

def print_shape(damage_dict):
	if damage_dict is None:
		print("Empty")
	else:
		tab = PrettyTable(["TYPE",'SHAPE'])
		for key in damage_dict:
			tab.add_row([key,damage_dict[key].shape])
		print(tab)

def flip_rois(p):
	#horizontally flip the detected bounding boxes.
	print("Flipping the rois")
	height,width = p['masks'].shape[1:]
	for i in range(len(p['scores'])):
		roi = list(p['rois'][i])
		box_w = roi[2]-roi[0]
		roi[0] = width-roi[0]
		roi[2] = width-roi[2]
		roi[0] -= box_w
		roi[2] += box_w
		p['rois'][i] = np.array(roi)
	return p

def process_damage_dict_tflite(damage_dict,limit_dict):
	final_rois = []
	final_masks = []
	final_scores = []
	predicted_class_names = []

	for i in range(len(damage_dict['class_labels'])):
		if limit_dict['panel'] >= damage_dict['scores'][i]:
			continue
		final_rois.append(damage_dict['rois'][i])
		if 'masks' in damage_dict:
			final_masks.append(damage_dict['masks'][i,:,:])
		final_scores.append(damage_dict['scores'][i])
		predicted_class_names.append(damage_dict['class_labels'][i])
	print("Filtered {} damages according to threshold".format(len(final_rois)))
	return final_rois,final_scores,final_masks,predicted_class_names

def process_damage_dict(damage_dict,damage_list,limit_dict):
	final_class_id = []
	final_rois = []
	final_masks = []
	final_scores = []
	predicted_class_names = []

	for i in range(len(damage_dict['class_ids'])):
		if limit_dict['panel'] >= damage_dict['scores'][i]:
			continue
		final_class_id.append(damage_dict['class_ids'][i])
		final_rois.append(damage_dict['rois'][i])
		if 'masks' in damage_dict:
			final_masks.append(damage_dict['masks'][i,:,:])
		final_scores.append(damage_dict['scores'][i])
		predicted_class_names.append(damage_list[damage_dict['class_ids'][i]])
	print("Filtered {} damages according to threshold".format(len(final_rois)))
	return final_rois,final_scores,final_class_id,final_masks,predicted_class_names

def remove_scratch_overlap_dent(roi_list,class_id_list,scores_list,masks_list,predicted_class_list):
	# This module removes the scratches that overlap with dents
	final_roi = []
	final_score = []
	final_class_id= []
	final_masks = []
	final_predicted_class= []
	duplicate_list = [False for i in range(len(roi_list))]

	for i in range(len(roi_list)):
		if predicted_class_list[i] == 'scratch':
			contour = mask_contour(masks_list[i])
			try:
				scratch_poly = Polygon(contour[0])
			except: continue
			for j in range(len(roi_list)):
				if predicted_class_list[j] == 'd2':
					contour = mask_contour(masks_list[j])
					try:
						d2_poly = Polygon(contour[0])
					except: 
						continue

					if scratch_poly.intersects(d2_poly):
						print(scratch_poly.intersection(d2_poly).area/scratch_poly.area)
						if scratch_poly.intersection(d2_poly).area/scratch_poly.area > 0.75:
							duplicate_list[i] = True

	for i in range(len(roi_list)):
		if not duplicate_list[i]:
			final_roi.append(roi_list[i])
			final_score.append(scores_list[i])
			final_class_id.append(class_id_list[i])
			final_masks.append(masks_list[i])
			final_predicted_class.append(predicted_class_list[i])
	print("Damages returned after overlapping scratch removal: ",len(final_roi),flush=True)
	return final_roi,final_score,final_class_id,final_masks,final_predicted_class





def remove_duplicate_damages(roi_list,class_id_list,scores_list,masks_list,predicted_class_list):
	# As we are doing multiple inference in generic damages, so we need to remove the duplicate detections.
	final_roi = []
	final_score = []
	final_class_id= []
	final_masks = []
	final_predicted_class= []
	duplicate_list = [False for i in range(len(roi_list))]
	# print("Predicted class names:")
	# for i in range(len(predicted_class_list)):
	# 	print(i, predicted_class_list[i])
	# print("Damages recieved in duplicate removal: ",len(roi_list),flush=True)

	for i in range(len(roi_list)):
		if not duplicate_list[i]:
			contour = mask_contour(masks_list[i])
			try:
				d1_poly = Polygon(contour[0])
			except: continue
			for j in range(i+1,len(roi_list)):
				if not duplicate_list[j]:
					if class_id_list[i] == class_id_list[j]:
						contour2 = mask_contour(masks_list[j])
						try:
							d2_poly = Polygon(contour2[0])
						except:continue
						if d1_poly.intersects(d2_poly):
							iou = mask_overlap.IntersectionOverUnion(d1_poly,d2_poly)
							if iou >= 0.5:
								# print("Damage: {}-{}-{} and Damage: {}-{}-{} are duplicate".format(i,predicted_class_list[i],scores_list[i],j,predicted_class_list[j],scores_list[j]))
								if scores_list[i] > scores_list[j]:
									duplicate_list[j] = True
								else:
									duplicate_list[i]  = True
	
	# print("No of duplicate found: ",sum(duplicate_list),flush=True)
	for i in range(len(roi_list)):
		if not duplicate_list[i]:
			final_roi.append(roi_list[i])
			final_score.append(scores_list[i])
			final_class_id.append(class_id_list[i])
			final_masks.append(masks_list[i])
			final_predicted_class.append(predicted_class_list[i])
	print("Damages returned after duplicate removal: ",len(final_roi),flush=True)
	return final_roi,final_score,final_class_id,final_masks,final_predicted_class


# some times the model detects some small damages inside a damage, but the detections are so small that they have less iou than thershold
# so if there is a false positive that is of same class and is inside another true positive,or intersects with a true positive, then 
# this part of the code will remove that false positive, as we already do overlapping damages merge in masking logic in prod.


def count_false_positive(image_stats):
	count = 0
	for ent in image_stats:
		if ent['status'] == 'false-positive':
			count += 1
	print("RETURNING COUNT",count)
	return count


def remove_wrong_false_positive(image_stats):
	# print("No of false positives initially: {}".format(count_false_positive(image_stats)))
	for i in range(len(image_stats)):
		if image_stats[i]['status'] == 'false-positive':
			try:
				poly_i = Polygon(image_stats[i]['pred_mask'][0])
			except:
				continue
			for j in range(len(image_stats)):
				if image_stats[i]['predicted_class'] == image_stats[j]['predicted_class'] and image_stats[j]['status']=='correctly-identified':
					# print("False positive at {} matching with correclty identified at {}".format(i,j))
					try:
						poly_j = Polygon(image_stats[j]['pred_mask'][0])
					except:
						continue
					if poly_i.intersects(poly_j):
						# print("They intersencts , changing false positive")
						#change the status of false-positive in i into correctly-identified.
						image_stats[i]['status']       = 'correctly-identified'
						image_stats[i]['gtref']        = 	image_stats[j]['gtref']
						image_stats[i]['actual_class'] = 	image_stats[j]['actual_class']
						image_stats[i]['gt_bb']        = 	image_stats[j]['gt_bb']
						image_stats[i]['gt_mask']      = 	image_stats[j]['gt_mask']
						break
	# print("No of false positives finally: {}".format(count_false_positive(image_stats)))

	return image_stats

def remove_Multipoly(multipoly): 
    """
            This function takes a multipoly and returns the polygon of maximum
            area from the multipoly.
            Arguments: multipoly: A shapely MultiPolygon object.
            Returns : a shapely Polygon object.
    """
    max_area = 0
    max_area_poly = None
    
    for poly in multipoly:
        if poly.area > max_area:
            max_area = poly.area
            max_area_poly = poly
    
    return max_area_poly


def divide_damages_by_size(gt_list, pred_mask,pred_classes,classes_to_divide,division_threshold):
	# process gt annotation first.
	print("Dividing damages by size")
	new_gt_damages  = []
	for region in gt_list:
		region_identity = region['region_attributes']['identity']
		if region_identity not in classes_to_divide:
			new_gt_damages.append(region)
			continue
		xlist = region['shape_attributes']['all_points_x']
		ylist = region['shape_attributes']['all_points_y']
		coords = zip(xlist,ylist)
		region_poly = Polygon(coords)
		if region_poly.area >= division_threshold[region_identity]['large']:
			region['region_attributes']['identity'] = region_identity + "_L"
		elif region_poly.area >= division_threshold[region_identity]['medium']:
			region['region_attributes']['identity'] = region_identity + "_M"
		else:
			region['region_attributes']['identity'] = region_identity + "_S"
		new_gt_damages.append(region)
	
	# now process predictions.
	for i in range(len(pred_mask)):
		if pred_classes[i] not in classes_to_divide:
			continue
		damage_contour = mask_contour(pred_mask[i])
		dam_poly = Polygon(damage_contour[0])
		if dam_poly.geom_type == 'MultPolygon':
			dam_poly = remove_Multipoly(dam_poly)
		if dam_poly.area >= division_threshold[pred_classes[i]]['large']:
			pred_classes[i] = pred_classes[i] + "_L"
		elif dam_poly.area >= division_threshold[pred_classes[i]]['medium']:
			pred_classes[i] = pred_classes[i] + "_M"
		else:
			pred_classes[i] = pred_classes[i] + "_S"
	return new_gt_damages, pred_classes

def fix_size_difference(gt_class,predicted_class,classes_to_divide):
	# print("fixing size differnces for ", gt_class, predicted_class)
	if any(i in gt_class for i in classes_to_divide):
		base_gt = gt_class.split('_')[0]
	else:
		base_gt = gt_class
	if any(i in predicted_class for i in classes_to_divide):
		base_pred = predicted_class.split('_')[0]
	else:
		base_pred = predicted_class
	if base_gt == base_pred:
		# print(f"Returning {predicted_class}")
		return predicted_class
	# print(f"Returning {gt_class}")
	return gt_class


def filter_gt_damages(gt_damages_list, panels_dict,panel_list=None,panels_in_view=None,translate_point=None,use_view=False): # only keep those damages that overlap with panels.
	print("filtering gt damages")
	inters = {}
	new_gt_damages  = []
	index_list = [False for i in range(len(gt_damages_list))]
	for di,region in enumerate(gt_damages_list):
		xlist = region['shape_attributes']['all_points_x']
		ylist = region['shape_attributes']['all_points_y']
		coords = zip(xlist,ylist)
		region_poly = Polygon(coords)
		if not region_poly.is_valid:
			region_poly=region_poly.convex_hull
		
		for i in range(len(panels_dict['scores'])):
			detected_panel = panel_list[panels_dict['class_ids'][i]]
			if use_view and detected_panel not in panels_in_view:
				# ignore this panel as this panel is not in the current view.
				continue
			mask = panels_dict['masks'][i,:,:]
			contour = mask_contour(mask)
			try:
				panel_poly = Polygon(np.fliplr(np.array(contour[0]))) # polygon coordinates flipped to x,y as gt is in xy.
			except Exception as e:
				print(e)
				continue
			if not panel_poly.is_valid:
				panel_poly = panel_poly.convex_hull
			if translate_point is not None: # shift pred panels to compensate for cropping.
				panel_poly = translate(panel_poly,xoff=translate_point[0],yoff=translate_point[1])
			if region_poly.intersects(panel_poly) and region_poly.intersection(panel_poly).area / region_poly.area > 0.05 : # don't consider the damage if damage and panel overlap is less than 5%
				index_list[di] = True

			
	for i in range(len(gt_damages_list)):
		if index_list[i]:
			new_gt_damages.append(gt_damages_list[i])
	print("Removed {} damages that don't overlap with predicted panels".format(len(gt_damages_list) - len(new_gt_damages)))
	# print(inters)
	# with open('over.json','w+') as f:
	# 	json.dump(inters,f)
	return new_gt_damages

def filter_pred_damages(panels_dict,final_rois,final_scores,final_class_ids,mid_masks,predicted_class_names,panel_list=None,panels_in_view=None,use_view=False):
	print("Filter pred damages")
	new_rois = []
	new_scores = []
	new_class_ids = []
	new_masks = []
	new_predicted_class_names = []
	index_list = [False for i in range(len(final_rois))]
	for di in range(len(final_scores)):
		dam_contour = mask_contour(mid_masks[di])
		try:
			dam_poly = Polygon(dam_contour[0])
		except:
			continue
		if dam_poly.geom_type == 'MultPolygon':
			dam_poly = remove_Multipoly(dam_poly)
		for pi in range(len(panels_dict['scores'])):
			detected_panel = panel_list[panels_dict['class_ids'][pi]]
			if use_view and detected_panel not in panels_in_view:
				# ignore this panel as this panel is not in the current view.
				continue
			mask = panels_dict['masks'][pi,:,:]
			contour = mask_contour(mask)
			try:
				panel_poly = Polygon(contour[0])
			except Exception as e:
				print(e)
				continue
			if dam_poly.intersects(panel_poly):
				index_list[di] = True

	for i in range(len(final_rois)):
		if index_list[i]:
			new_rois.append(final_rois[i])
			new_scores.append(final_scores[i])
			new_class_ids.append(final_class_ids[i])
			new_masks.append(mid_masks[i])
			new_predicted_class_names.append(predicted_class_names[i])

	print("Damages remained after removing non-panel damages ", len(new_rois))
	return new_rois,new_scores,new_class_ids,new_masks,new_predicted_class_names

def masks_adjustment(detected_masks, image_template, shiftu):
	print("SHIFT POINTS (x,y): ", shiftu, flush=True)
	add_x = shiftu[0]
	add_y = shiftu[1]
	
	new_masks = []
	for i in range(0, len(detected_masks)):
		new_masks.append(copy.deepcopy(image_template))
  
	for m, each_mask in enumerate(new_masks):
		each_mask[add_y : add_y + detected_masks[m].shape[0], add_x : add_x + detected_masks[m].shape[1]] = detected_masks[m]

	new_masks = np.array(new_masks)
	return new_masks
			

def get_view_panles(panels_dict,panels_list):
	detected_panels = [panels_list[i] for i in panels_dict['class_ids']]




	
 



