import os
import json
import sys

direc = sys.argv[1]

with open(os.path.join(direc,'via_region_data.json')) as f:
    annot = json.load(f)

damages  ={k:0 for k in ['scratch','d1','d2','d3','bumperdent','broken','clipsbroken','shattered','tear','bumpertear']}
for key in annot:
    for region in annot[key]['regions']:
        region_name = region['region_attributes']['identity']
        if region_name in damages:
            damages[region_name] += 1
        else:
            damages[region_name] = damages.get(region_name,0)+1

print(damages)
