
import sys
import os
from PIL import Image
from tqdm.notebook import tqdm
from test_utils.predict_yolo import predict_yolo
from test_utils.predict_tflite import load_tflite_model,  predict_tflite
from test_utils.testing_utils import calculate_metrics
from test_utils.predict_det2 import detectron_prediction, load_dirt, process_damage_dict, process_det2_dict

device = "cpu"
sys.path.append("..")
sys.path.append("../..")
from utils.data_utils import read_img, read_json, dump_json, draw_box, write_text
#from ultralytics import YOLO
from collections import defaultdict, Counter


class_list = ['dirt', 'bird_dropping']
#Note: The fuel_level in class list above is actual fuelgauge. During training, it was called fuel_level

def compute_metrics(class_name, thresh_to_test, iteration, gtdict, imgdir, dest_dir, 
                    save_images=False,save_metrics_text=False, iou_threshold=0.5):
    print(f"Results for {class_name}")

    for thresh in thresh_to_test:
        c = 0
        #conf_thresholds = {k:thresh for k in class_list}
        #thresh_list = "_".join([str(i) for i in list(conf_thresholds.values())])
        thresh_list = f"{thresh}" # single threshold in image name
        resdict = read_json(f"{dest_dir}/result_jsons/{iteration}_results_{thresh_list}.json")
        total_tps = total_fps = total_fns = 0
        for imgname, gt_list in gtdict.items():
            #if imgname != "iowNQDRsov_1713348304039.jpg":continue
            c += 1
            folder_name = annot_file[imgname]['filename'].split("/")[0]
        
            
            pred = resdict[imgname]
            gt1 = [d for d in gt_list if d['class'] == class_name]
            
            pred = [d for d in pred if d['class'] == class_name]
            
            gt = gt1


            #pred = [d for d in pred if d['class'] == pred_name]
            
            # remove duplicate gts
            # gt = []
            # for i in gt1:
            #     if i not in gt:
            #         gt.append(i)
                    
            tps, fps, fns = calculate_metrics(gt, pred, iou_threshold = iou_threshold)

            if save_images:
                #read img
                imgpath = f"{imgdir}/{imgname}"
                img = read_img(imgpath)
                # draw boxes:
                for g in gt:
                    identity = g['class']
                    if identity == 'fuel_level':identity ='fuelgauge'
                    bbox = g['bbox']
                    img = draw_box(img, bbox, (0,255,0)) # use polygon here
                    img = write_text(img, identity, bbox[:2], color = (0,255,0))
                    
                for p in pred:
                    pred_class = p['class']
                    bbox = p['bbox']
                    img = draw_box(img, bbox, (255,0,0))
                    img = write_text(img, pred_class, bbox[:2], color = (255,0,0))
                    
                #result_img_dir = f"{dest_dir}/error_analysis_{thresh_list}/{class_name}_{iteration}" # Results will be stored in this folder 
                result_img_dir = f"{dest_dir}/{class_name}_{thresh_list}"
                if fps > 0:
                    destpath = f"{result_img_dir}/FP"#_{thresh_list}"
                    os.makedirs(destpath ,exist_ok=True)
                    Image.fromarray(img).save(f"{destpath}/{imgname}")
                    
                if fns>0:
                    destpath = f"{result_img_dir}/FN"#_{thresh_list}"
                    os.makedirs(destpath ,exist_ok=True)
                    Image.fromarray(img).save(f"{destpath}/{imgname}")
                
                if tps > 0:
                    destpath = f"{result_img_dir}/TP"#_{thresh_list}"
                    os.makedirs(destpath ,exist_ok=True)
                    Image.fromarray(img).save(f"{destpath}/{imgname}")
                    
                
            
            total_tps += tps
            total_fps += fps
            total_fns += fns
            #print(total_fns, total_fps, total_tps)
            '''
            if fps < 0:
                print(imgname)
                print("tp: ", tps)
                print("fps: ", fps)
                print("fns: ", fns)
                
                print('gt: ', gt)
                print('pred: ', pred)
            '''
        
        
        try:
            txt_to_write = f"""Threshold: {thresh}\nPrecision for {class_name}:  {(total_tps)/(total_tps+total_fps)}\nRecall for {class_name}: {(total_tps)/(total_tps+total_fns)}\nTrue Positives: {total_tps}\nFalse Positives: {total_fps}\nFalse Negatives: {total_fns}\n =====================================================\n"""
        except Exception as e:
            txt_to_write = f"""Threshold: {thresh}\nError: {e}\nTrue Positives: {total_tps}\nFalse Positives: {total_fps}\nFalse Negatives: {total_fns}\n=====================================================\n"""
                           
                           
        print(txt_to_write)
        if save_metrics_text:  
            os.makedirs(f"{dest_dir}/metrics_texts", exist_ok=True)                   
            with open(f"{dest_dir}/metrics_texts/{class_name}.txt", "a") as f:
                f.write(txt_to_write)
                print(f"Wrote metrics for {class_name} to {dest_dir}/metrics_texts/{class_name}.txt")
        




def get_yolo_results(iteration,project, conf_threshold_list, imgdir,imgsize, class_list, dest_dir, save_json_to_disk = False):
    model_path = f"yolo_runs/{project}/{iteration}/weights/best.pt"
    model = YOLO(model_path).to(device) #Load Model
    print(f"Loaded to {model.device}")
    print(f"Running model {iteration} with imgsize {imgsize} ================================ ")
    
    dest_dir = f"{dest_dir}/result_jsons"
    
    for conf_thresh in conf_threshold_list:
        conf_thresholds = {k:conf_thresh for k in class_list}
        print(f"Processing for threshold {list(conf_thresholds.values())[0]}")
        resdict = predict_yolo(imgdir, model, imgsize,
                        iteration, class_list, dest_dir, 
                        conf_thresholds, iou_nms_thresh =0.2, save_results=save_json_to_disk, use_mps = False)
        
        print("===============================================")


def get_det2_results(imgdir, class_list, conf_threshold_list, save_json_to_disk = False):
    predictor = load_dirt()

    results_dict = {}

    for conf_thresh in conf_threshold_list:
        for imgname in os.listdir(imgdir):
            if imgname.endswith(('json', 'Store', 'pkl')):continue

            imgpath = f"{imgdir}/{imgname}"

            test_image = read_img(imgpath)

            detection_dict = detectron_prediction(predictor, test_image)

            detection_list = process_det2_dict(detection_dict, class_list, conf_threshold=conf_thresh)

            results_dict[imgname] = detection_list


        if save_json_to_disk:
            os.makedirs(f"{dest_dir}", exist_ok=True)
            dump_json(results_dict, f"{dest_dir}/{iteration}_results_{conf_thresh}.json", indent = 1)
            print(f"json saved for thresh {conf_thresh}")






if __name__ == '__main__':
    
    use_model = 'yolo' # yolo, tflite

    generate_results = False
    if generate_results:
        #HPs ===========================================
        conf_threshold_list = [0.1]#[0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9] #thresholds to generate results for

        dtype = "Ikea" # test, Celerity, Ikea, Flow Logistics, Warrior Logistics 
        imgsize = 1024
        iteration = f"v1_p_n_{imgsize}" #will be used for model path
        
        
        yolo_dataset_dir = "ic_yolo_datasetv1"
        
        save_json_to_disk = True # saves results dict to disk
        #HPs ===========================================
        
        #imgdir = f"datasets/{yolo_dataset_dir}/{dtype}/images" #YOLO images # does not include empty annotations
        #imgdir = "test_datasets/Project_Odometer-OCR/combined"
        #imgdir = "test_datasets/Project Mahindra -Instrument Panel Test Data/combined"
        imgdir = f"test_datasets/Project {dtype} Instrument Cluster/combined" # EU VAN DEALERS
        
        if 'odo' in imgdir.lower():
            dest_dir = f"results/instrumentpanel/{iteration}/project_odometer"
        elif 'instrument' in imgdir.lower() and 'combined' in imgdir.lower() and 'Cluster' not in imgdir:
            dest_dir = f"results/instrumentpanel/{iteration}/project_instru_testdata"
        elif 'Cluster' in imgdir:
            dest_dir = f"results/instrumentpanel/{iteration}/{dtype}"
        else:
            dest_dir = f"results/instrumentpanel/{iteration}/instrument_panel" #part of train data

        project = "instrument_panel"  # this is used for model path

        print(f"Saving Results to: {dest_dir}\n")

        #conf_thresholds = {k:0.1 for k in class_list} # add default 0.1 thresh to all classes
        #conf_thresholds = read_json("thresholds_dict.json")

        if use_model == 'tflite':
            dest_dir= f"{'/'.join(dest_dir.split('/')[:-1])}/tflite/{dest_dir.split('/')[-1]}" #f"results/{iteration}/tflite/instrument_panel"
            get_tflite_results(iteration, project, conf_threshold_list, imgdir, imgsize, class_list, dest_dir, save_json_to_disk = save_json_to_disk)
        else:
            get_yolo_results(iteration, project, conf_threshold_list, imgdir, imgsize, class_list, dest_dir, save_json_to_disk = save_json_to_disk)
        
        
    get_metrics = True
    if get_metrics:
        #HPs =====================================================================
        dtype = "Celerity" # test, Celerity, Ikea, Flow Logistics, Warrior Logistics 
        imgsize = 1024
        iteration = f"v1_p_n_{imgsize}"
        thresh_to_test = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        thresh_to_test = [0.3]
        class_to_inspect = 'opendoor_warning' # which class scores to inspect. Keep this empty if all class metrics to be inspected once and stored 
        save_images = True #Save FP,TP and FN images annotated with GT and predictions
        save_metrics_text = False # If metrics for all thresholds of all classes to be written to text file.
        iou_threshold = 0.4 #between gt and pred
        # ===================================================================================
        
        
        #vgg_data_dir = "Project_Mahindra_Interior-Instrument_Panel"
        #vgg_data_dir = "Project Mahindra -Instrument Panel Test Data"
        #vgg_data_dir = "Project_Odometer-OCR"
        vgg_data_dir = f"Project {dtype} Instrument Cluster" # EU VAN DEALERS
        
        yolo_data_dir = "ic_yolo_datasetv1"
        
        
        if 'odo' in vgg_data_dir.lower():
            imgdir = f"test_datasets/{vgg_data_dir}/combined"
            gtdict = read_json(f"test_datasets/{vgg_data_dir}/gtdict.json")
            annot_file = read_json(f"test_datasets/{vgg_data_dir}/via_region_data.json")
            dest_dir = f"results/instrumentpanel/{iteration}/project_odometer"
        elif 'instrument' in vgg_data_dir.lower() and 'test' in vgg_data_dir.lower():
            imgdir = f"test_datasets/{vgg_data_dir}/combined"
            gtdict = read_json(f"test_datasets/{vgg_data_dir}/gtdict.json")
            annot_file = read_json(f"test_datasets/{vgg_data_dir}/via_region_data.json")
            dest_dir = f"results/instrumentpanel/{iteration}/project_instru_testdata"
        elif 'Cluster' in vgg_data_dir:
            imgdir = f"test_datasets/{vgg_data_dir}/combined"
            gtdict = read_json(f"test_datasets/{vgg_data_dir}/gtdict.json")
            annot_file = read_json(f"test_datasets/{vgg_data_dir}/via_region_data.json")
            dest_dir = f"results/instrumentpanel/{iteration}/{dtype}"
        else:
            imgdir = f"datasets/{yolo_data_dir}/{dtype}/images"
            gtdict = read_json(f"datasets/{vgg_data_dir}/{dtype}/{dtype}_gtdict.json")
            annot_file = read_json(f"datasets/{vgg_data_dir}/{dtype}/via_region_data.json")
            dest_dir = f"results/instrumentpanel/{iteration}/instrument_panel"

        if use_model == 'tflite':
            dest_dir= f"{'/'.join(dest_dir.split('/')[:-1])}/tflite/{dest_dir.split('/')[-1]}" #f"results/{iteration}/tflite/instrument_panel"

    
        for class_name in class_list:
            if class_to_inspect != "" and class_name != class_to_inspect:continue # if a class name is added to class_to_inspect, only that class is inspected.
            
            compute_metrics(class_name, thresh_to_test, iteration, gtdict, imgdir, dest_dir, save_images=save_images,save_metrics_text=save_metrics_text, iou_threshold=iou_threshold)