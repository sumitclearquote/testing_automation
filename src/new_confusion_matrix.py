import pandas as pd
import json
import os


def confusion_matrix(make,model,threshold,damage_list,path,dest,graph_dict,csv):
    df = pd.read_csv(path)
    #if(make == '' and weights_type ==)
    class_count = {}

    for damage in damage_list:
        # print(damage,flush=True)
        temp = df[df['actual_class']==damage]
        class_count[damage] = len(set(zip(temp['gtref'],temp['S.NO'])))
    print('Class Count:',class_count)
    actual_pred = {}

    for k in sorted(damage_list):
        actual_pred[k] = 0

    cm = pd.DataFrame(actual_pred,index=list(actual_pred.keys()))
    #cm[col][row]
    #print(cm,flush=True)
    
    l = 0 #lower bound
    # for every S.NO i.e for every image get its df
    print('IMAGES',len(set(df['IMAGE'])))
    sn_list = list(df['S.NO'].unique())  #taking each unique image and it's damages
    for i in sn_list:
        d = df[df['S.NO'] == i]  #dataframe of each image (can have muliple damages)


        #print(d)Panel is
        flag_gtref = {} # to count same TP gtref once
        fn_list = [] # to count same FN gtref once
        #print(d,flush=True)
        for k in set(d['gtref']):
            #if str(k) != 'nan':
            flag_gtref[k] = False
        # print(flag_gtref,flush=True)

        u = l + len(d) #upper bound
        #print('LOWER',l)
        #print('UPPER',u)
        for j in range(l,u):

                #edit
            # if not(d._get_value(j,'IMAGE') in os.listdir('sample_set')):
            #   break

            # print(d._get_value(j,'IMAGE'),flush=True)
            gtref_val = d._get_value(j,'gtref')

            if(d._get_value(j,'status') == 'correctly-identified' and not(flag_gtref[gtref_val])):

                if(d._get_value(j,'confidence') >= threshold): #threshold
                    flag_gtref[d._get_value(j,'gtref')] = True # class of gtref counted
                    cm[d._get_value(j,'actual_class')][d._get_value(j,'predicted_class')] += 1 #TP
                else:
                    if not(d._get_value(j,'gtref') in fn_list):
                        fn_list.append(d._get_value(j,'gtref'))
                        cm[d._get_value(j,'actual_class')]['no-damage'] += 1 #FN


            elif(d._get_value(j,'status') == 'wrong-detection' and d._get_value(j,'confidence') >= threshold):
                try:
                    if(flag_gtref[d._get_value(j,'gtref')] == True ):
                        cm['no-damage'][d._get_value(j,'predicted_class')] += 1 # only FP
                    else:
                        cm[d._get_value(j,'actual_class')]['no-damage'] += 1  #FN
                        cm['no-damage'][d._get_value(j,'predicted_class')] += 1 #FP
                except:
                    continue

            elif(d._get_value(j,'status') == 'false-positive' and d._get_value(j,'confidence') >= threshold):
                cm['no-damage'][d._get_value(j,'predicted_class')] += 1

            elif(d._get_value(j,'status') == 'non-detection'):
                try:
                    cm[d._get_value(j,'actual_class')]['no-damage'] += 1  #FN
                except:
                    continue
        l = u
        #break

    cm.fillna(0)


    for k in damage_list:
        cm[k]['no-damage'] = class_count[k] - cm[k][k]


    cm_path = os.path.join(dest,'matrix_'+csv)
    # cm_path = os.path.join(dest,'cm_old_crop.csv')
    with open(cm_path,'a') as f:
        f.write('\n'+str(threshold)+"\n")

    cm.to_csv(cm_path,mode='a')
    precision = {}
    recall = {}
    for k in damage_list:
        tp = cm[k][k]
        tp_fp = tp + cm['no-damage'][k] # FP
        precision[k] = tp / tp_fp

    for k in damage_list:
        tp = cm[k][k]
        tp_fn = tp + cm[k]['no-damage'] # FN
        recall[k] = tp / tp_fn

    print(cm,flush=True)
    print('Precision:',precision)
    print('Recall:',recall)
    f1_score = {}
    for k in damage_list:
        try:
            f1_score[k] = (2*(precision[k]*recall[k]))/(precision[k]+recall[k])
        except:
            f1_score[k] = 'NaN'


    graph_dict[threshold] = {"Precision":precision,"Recall":recall,"F1":f1_score}
    f = open(cm_path,'a')
    f.write('\n,'+'Precision,'+'Recall,'+'F1-Score\n')
    for k in damage_list:
        f.write(str(k)+','+str(precision[k])+','+str(recall[k])+','+str(f1_score[k])+'\n')
    return graph_dict


if __name__ == '__main__':
    make = ''
    model = ''
 
    # damage_list = ['scratch_S','scratch_M','scratch_L', 'd2_S','d2_M','d2_L', 'tear', 'clipsbroken',  'shattered', 'broken','no-damage']
    
    #scract+dent combined model
    # damage_list = ['rust','paintedrust','no-damage']
    damage_list = ['dirt','no-damage']


 
    #dest_path = '../testing/dirt_V1/'
    dest_path = "../testing/dirt_V2/"
    for csv in os.listdir(dest_path):
        path_to_csv = dest_path
        if csv.endswith('csv') and 'matrix' not in csv:
            path_to_csv = os.path.join(path_to_csv,csv)
        else:
            continue
        if os.path.exists(os.path.join(dest_path,'matrix_'+csv)):
            continue
        generate_for = [0.1, 0.15, 0.20, 0.25,0.30,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,0.99]
        # generate_for = [0.1]
        graph_dict = {}
        for threshold in generate_for:
            graph_dict=confusion_matrix(make, model, threshold, damage_list, path_to_csv, dest_path,graph_dict,csv)
        # break
    
    # with open(os.path.join(dest_path,'debug_full.json'),'w') as f:
    #   json.dump(graph_dict,f)
 
 
 
 
 
 
 #ExtraPanels
    # damage_list = ['alloywheel', 'tyre', 'fuelcap', 'headlightwasher', 'leftapillar', 'leftbpillar', 'leftcpillar', 'leftdpillar',
    #   'leftfrontdoorcladding', 'leftfrontdoorglass', 'leftfrontrocker', 'leftfrontventglass', 'leftquarterglass',
    #   'leftreardoorcladding', 'leftreardoorglass', 'leftrearrocker', 'leftrearventglass', 'leftroofside', 'licenseplate',
    #   'logo', 'lowerbumpergrille', 'namebadge', 'Reflector', 'rightapillar', 'rightbpillar', 'rightcpillar',
    #   'rightdpillar', 'rightfrontdoorcladding', 'rightfrontdoorglass', 'rightfrontrocker', 'rightfrontventglass',
    #   'rightquarterglass', 'rightreardoorcladding', 'rightreardoorglass', 'rightrearrocker', 'rightrearventglass',
    #   'rightroofside', 'Roof', 'sensor', 'towbarcover', 'variant', 'wheelcap', 'wheelrim', 'sunroof',
    #   'frontbumpercladding', 'rearbumpercladding', 'wiper', 'indicator', 'roofrail', 'frontbumpergrille','no-damage']
    
    #VansPanels
    # damage_list = ['alloywheel', 'bonnet', 'doorglass', 'doorhandle', 'footstep', 'frontbumper', 'frontbumperfiller', 'frontbumpergrille', 
    #   'frontws', 'hinge', 'leftbumperfiller', 'leftfender', 'leftfoglamp', 'leftfrontdoor', 'leftheadlamp', 'leftorvm', 'leftqpanel',
    #   'leftrearcorner', 'leftrunningboard', 'leftsidefiller', 'leftslidingdoor', 'lefttailgate', 'lefttaillamp', 'leftvalence', 'leftwa',
    #   'rearbumper', 'rearws', 'rightbumperfiller', 'rightfender', 'rightfoglamp', 'rightfrontdoor', 'rightheadlamp', 'rightorvm', 
    #   'rightqpanel', 'rightrearcorner', 'rightrunningboard', 'rightsidefiller', 'rightslidingdoor', 'righttailgate', 'righttaillamp',
    #   'rightvalence', 'rightwa', 'tailgate', 'tyre', 'wheelcap', 'wheelrim', 'leftrearcover', 'rightrearcover', 'no-damage']

    # damage_list=['bonnet', 'doorhandle', 'frontbumper', 'frontws', 'leftbootlamp', 'leftfender',
                # 'leftfoglamp', 'leftfrontdoor', 'leftheadlamp', 'leftorvm', 'leftqpanel', 'leftreardoor',
                # 'leftrunningboard', 'lefttaillamp', 'leftvalence', 'leftwa', 'mtailgate', 'rearbumper',
                # 'rearws', 'rightbootlamp', 'rightfender', 'rightfoglamp', 'rightfrontdoor', 'rightheadlamp',
                # 'rightorvm', 'rightqpanel', 'rightreardoor', 'rightrunningboard', 'righttaillamp', 'rightvalence',
                # 'rightwa', 'tailgate', 'footstep','no-damage']
    #damage_list = ['scratch','d2','no-damage']
    
