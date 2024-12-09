
import csv
import json
import pandas as pd

def write_to_csv(d,path_to_csv):
	#j = json.load(open('result/tested.json'))
	jsn = d
	#jsn = j
	for key in jsn.keys():
		for i in jsn[key]:
			if('pred_bb' in i.keys()):
				del i['pred_bb']
			if('pred_mask' in i.keys()):
				del i['pred_mask']
			if('gt_bb' in i.keys()):
				del i['gt_bb']
			if('gt_mask' in i.keys()):
				del i['gt_mask']
	f = open(path_to_csv,'w')
	writer = csv.writer(f)

	writer.writerow(['IMAGE','S.NO','gtref','status','IoU','confidence','actual_class','predicted_class'])

	c = 1
	for key in jsn.keys():
		for d in jsn[key]:
			writer.writerow([ key, c, d['gtref'], d['status'], d['IoU'], d['confidence'], d['actual_class'], d['predicted_class'] ])
		c+=1

	f.close()
	print('Written to csv...',flush=True)
	
	# with open('res.json','w') as f:
	#  	f.write(json.dumps(jsn,indent=5))


	# 	for i in jsn[key]:
	# 		for j in i.keys():
	# 			s.add(j)
	#l = ['IoU', 'predicted_class', 'actual_class', 'confidence', 'status', 'class'] #list(s)
	#print(jsn)
	#print(l)


	#l.insert(0,'IMAGES')

	# for k in jsn.keys():
	# 	for i in jsn[k]:
	# 		for j in l:
	# 			if not(j in i.keys()):
	# 				i[j] = ""

def confusion_matrix(make,model,threshold,damage_list):
    	#df = pd.read_csv('result/5e79a6fa5026365e15eb3eff/test_results.csv')
	df = pd.read_csv('TwinnerJune/Twinprod.csv')
	damage_list = ['scratch', 'd2', 'tear', 'clipsbroken',  'shattered', 'broken','no-damage']
	#if(make == '' and weights_type ==)
	actual_pred = {}
	for k in sorted(damage_list):
		actual_pred[k] = 0

	cm = pd.DataFrame(actual_pred,index=list(actual_pred.keys()))
	#cm[col][row]
	#print(cm,flush=True)
	
	l = 0
	# for every S.NO i.e for every image get its df
	print('IMAGES',len(set(df['IMAGE'])))
	for i in range(len(set(df['IMAGE']))):
		print('IMAGE:',i,flush=True)
		d = df[df['S.NO']==i+1]
		#print(d)
		flag_gtref = {}
		#print(d,flush=True)
		for k in set(d['gtref']):
			#if str(k) != 'nan':
			flag_gtref[k] = False
		print(flag_gtref,flush=True)

		
		
		u = l + len(d)
		print('LOWER',l)
		print('UPPER',u)
		for j in range(l,u):
			gtref_val = d.get_value(j,'gtref')

			if(d.get_value(j,'status') == 'correctly-identified' and not(flag_gtref[gtref_val])):
				
				if(d.get_value(j,'confidence') >= 0.5): #threshold
					flag_gtref[d.get_value(j,'gtref')] = True # class of gtref counted
					cm[d.get_value(j,'predicted_class')][d.get_value(j,'predicted_class')] += 1 #TP
				else:
					cm[d.get_value(j,'actual_class')]['no-damage'] += 1 #FN

			elif(d.get_value(j,'status') == 'wrong-detection' and d.get_value(j,'confidence') >= 0.5):
				if(flag_gtref[d.get_value(j,'gtref')] == True ):
					cm['no-damage'][d.get_value(j,'predicted_class')] += 1 # only FP
				else:
					cm[d.get_value(j,'actual_class')]['no-damage'] += 1  #FN
					cm['no-damage'][d.get_value(j,'predicted_class')] += 1 #FP

			elif(d.get_value(j,'status') == 'false-positive' and d.get_value(j,'confidence') >= 0.5):
				cm['no-damage'][d.get_value(j,'predicted_class')] += 1

			elif(d.get_value(j,'status') == 'non-detection'):
				cm[d.get_value(j,'actual_class')]['no-damage'] += 1  #FN
		l = u

		#break
		# for j in flag_gtref.keys():
		# 	if not(flag_gtref[j]):
		# 		cm[d.get_value(j,'actual_class')]['no-damage'] += 1  #FN
	cm.to_csv('TwinProd_CM.csv')
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
	f = open('TwinProd_CM.csv','a')
	f.write('\n,'+'Precision,'+'Recall\n')
	for k in damage_list:
		f.write(str(k)+','+str(precision[k])+','+str(recall[k])+'\n' )