import numpy as np
import time

def crop_car(detection_model, img,expand_crop=False):
    print("Inside Cropping Module", flush=True)
    st = time.time()
    try:
        predictions = detection_model(img)
        print("Detection time:", time.time()-st, flush=True)
    except Exception as e:
        print("prediction failed.", flush=True)
        print(str(e), flush=True)
        return None, None
    car_dict = {}
    out = predictions["instances"].to("cpu")
    # print(out)
    # print(type(out))
    all_fields = out.get_fields()
    car_dict['class_ids'] = all_fields['pred_classes'].numpy()
    car_dict['rois'] = all_fields['pred_boxes'].tensor.numpy()
    car_dict['scores'] = all_fields['scores'].numpy()
    if len(car_dict['scores']) == 0:
        print("No car detected", flush=True)
        return None, None
    area_list = []
    for box in car_dict['rois']:
        box_area = abs((box[0] - box[2]) * (box[1] - box[3]))
        area_list.append(box_area)

    car_dict['area'] = np.array(area_list)
    # print(car_dict)
    biggest_car_ind = np.where(car_dict['area'] == np.max(car_dict['area']))[0]
    # print(biggest_car_index)

    biggest_car = {
        'class_ids': np.array([car_dict['class_ids'][biggest_car_ind]]),
        'scores': np.array([car_dict['scores'][biggest_car_ind]],
                           dtype=np.float32),
        'rois': np.array([car_dict['rois'][biggest_car_ind]],
                         dtype=np.float32),
    }
    print("Biggest Car Box:", biggest_car['rois'][0], flush=True)
    print("Image dimensions:", {'width': img.shape[1], 'height': img.shape[0]})
    print("Max Area Box:", np.max(car_dict['area']), flush=True)
    rat = np.max(car_dict['area'])/(img.shape[0]*img.shape[1])
    if rat > 0.8:  # bounding box covers more than 80% area, so no cropping.
        return None, None

    x1, y1, x2, y2 = biggest_car['rois'][0][0]
    x1, y1, x2, y2 = int(round(x1)), int(round(y1)),\
        int(round(x2)), int(round(y2))
    # increase the width of box by 10% on both left and right side.
    img_width = img.shape[1]
    if expand_crop:
        box_width = x2-x1
        print("Box width:", box_width, flush=True)
        box_width10 = int(0.10*box_width)
        print("Box width 10%:", box_width10)
        print("Original x1 and x2: ", x1, x2, flush=True)
        x1 -= box_width10
        x2 += box_width10
        print("Updated x1 and x2 ", x1, x2, flush=True)
        box_height = y2-y1
        print("Box height: ",box_height,flush=True)
        print("Width/Height: ",box_width/box_height,flush=True)
        # if  0.85 < (box_width/box_height) < 1.15: # box will be square (ratio 1) for front and rear views. No top crop for these images.
        #     y1 = 1
        y1 = 1
        y1 = max(y1,0)
        print("Modified y1:",y1,flush=True)
    if x1 < 0:
        x1 = 0
    if x2 > img_width:
        x2 = img_width-1 
    img_crop = img[y1:y2, x1:x2]
    # torch.cuda.empty_cache()
    return (img_crop, (x1, y1))