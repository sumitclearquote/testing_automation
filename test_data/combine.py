import os 
import json
import sys
import shutil
direc = sys.argv[1]
combine_json ={}
for f_id in os.listdir(direc):
	if f_id.endswith(('.py','json')): continue
	json_file = [i for i in os.listdir(os.path.join(direc,f_id)) if i.endswith('json')][0]
	json_path = os.path.join(direc,f_id,json_file)
	print(json_path)
	with open(json_path) as f:
		l_json = json.load(f)
	for j in l_json :
		combine_json[j] = l_json[j]
with open(os.path.join(direc,'via_region_data.json'),'w+') as f:
	json.dump(combine_json,f)

print("Now deleting inner jsons")
for f_id in os.listdir(direc):
	if f_id.endswith(('.py','json')): continue
	for file in os.listdir(os.path.join(direc,f_id)):
		if file.endswith('json'):
			os.remove(os.path.join(direc,f_id,file))
		else:
			os.rename(os.path.join(direc,f_id,file),os.path.join(direc,file))
	
	shutil.rmtree(os.path.join(direc,f_id))