def damage_classes_config(make,model,weights_type):

	damage_list = ['BG', 'scratch', 'd1', 'd2', 'd3', 'tear', 'fade', 'bumperdent', 'bumpertear', 'clipsbroken', 'bumpertorn', 'broken', 'shattered', 'cracked']

	if(make in ['ackouat', 'ackoprod'] and weights_type in ['keras', 'tf']):
		print("Running Acko Generic model.", flush = True)

		panel_limit = 0.90 # GenericAckoModel_58_85_S90_O85_P90_acko_data_19082019 # 0.95
		class_names = ['BG', 'bonnet', 'frontbumper', 'leftfender', 'leftfrontdoor', 'leftreardoor', 'leftrunningboard', 'leftqpanel', 'leftorvm', 'leftheadlamp', 'lefttaillamp',
		'leftfoglamp', 'leftwa', 'leftbootlamp', 'rightfender', 'rightfrontdoor', 'rightreardoor', 'rightrunningboard', 'rightqpanel', 'rightorvm', 'rightheadlamp', 'righttaillamp',
		'rightfoglamp', 'rightwa', 'rightbootlamp', 'rearbumper', 'tailgate', 'mtailgate', 'frontws', 'rearws', 'doorhandle', 'leftvalence', 'rightvalence']

        #damage_list = ['BG', 'scratch', 'd1', 'd2', 'd3', 'tear', 'fade', 'bumperdent', 'bumpertear', 'clipsbroken', 'bumpertorn', 'broken', 'shattered', 'cracked']


    # RR
	elif(make == "Land Rover" and model == "Range Rover"):
		print("Running Range Rover model.")
		panel_limit = 0.90

		if(weights_type == 'detectron2'):
			panel_list = ['BG', 'bonnet', 'rightfrontbumper', 'rightfoglamp', 'rightheadlamp', 'rightfender', 'rightfenderrocker', 'rightfrontdoor',
			'rightfrontrocker', 'rightreardoor', 'rightrearrocker', 'rightqpanel', 'righttaillamp', 'rightrearbumper', 'rightorvm', 'uppertailgate', 'lowertailgate', 'rearws',
			'leftfrontbumper','leftfoglamp','leftheadlamp', 'leftfender', 'leftfenderrocker', 'leftfrontdoor', 'leftfrontrocker', 'leftreardoor', 'leftrearrocker', 'leftqpanel',
			'lefttaillamp', 'leftrearbumper', 'leftorvm', 'frontws', 'doorhandle']

			damage_list = ['scratch', 'd2']

		else:
			panel_list = ['BG', 'bonnet', 'rightfrontbumper', 'rightfoglamp', 'rightheadlamp', 'rightfender', 'rightfenderrocker', 'rightfrontdoor',
			'rightfrontrocker', 'rightreardoor', 'rightrearrocker', 'rightqpanel', 'righttaillamp', 'rightrearbumper', 'rightorvm', 'uppertailgate', 'lowertailgate', 'rearws',
			'leftfrontbumper','leftfoglamp','leftheadlamp', 'leftfender', 'leftfenderrocker', 'leftfrontdoor', 'leftfrontrocker', 'leftreardoor', 'leftrearrocker', 'leftqpanel',
			'lefttaillamp', 'leftrearbumper', 'leftorvm', 'frontws', 'doorhandle']

			#damage_list = ['BG', 'scratch', 'd1', 'd2', 'd3', 'tear', 'fade', 'bumperdent', 'bumpertear', 'clipsbroken', 'bumpertorn', 'broken', 'shattered', 'cracked']

	#RRS
	elif(make == "Land Rover" and model == "Range Rover Sport"):
		print("Running Range Rover Sport model.")
		panel_limit = 0.90

		if(weights_type == 'detectron2'):
			panel_list = ['BG','bonnet', 'rightfrontbumper', 'rightfoglamp', 'rightheadlamp', 'rightfender', 'rightfenderrocker', 'rightfrontdoor',
			'rightfrontrocker', 'rightreardoor', 'rightrearrocker', 'rightqpanel', 'righttaillamp', 'rightrearbumper', 'rightrearbumperclading',
			'rightorvm', 'rightrearcorner', 'tailgate','rearws', 'leftfrontbumper', 'leftfoglamp', 'leftheadlamp', 'leftfender',
			'leftfenderrocker', 'leftfrontdoor', 'leftfrontrocker',"leftreardoor", "leftrearrocker", "leftqpanel", "lefttaillamp",
			'leftrearbumper', 'leftrearbumperclading', "leftorvm", 'leftrearcorner', "frontws", "doorhandle", 'reartowhookcover', 'fronttowhookcover']

			damage_list = ['scratch', 'd2']

		else:

			panel_list = ['BG','bonnet', 'rightfrontbumper', 'rightfoglamp', 'rightheadlamp', 'rightfender', 'rightfenderrocker', 'rightfrontdoor',
			'rightfrontrocker', 'rightreardoor', 'rightrearrocker', 'rightqpanel', 'righttaillamp', 'rightrearbumper', 'rightrearbumperclading',
			'rightorvm', 'rightrearcorner', 'tailgate','rearws', 'leftfrontbumper', 'leftfoglamp', 'leftheadlamp', 'leftfender',
			'leftfenderrocker', 'leftfrontdoor', 'leftfrontrocker',"leftreardoor", "leftrearrocker", "leftqpanel", "lefttaillamp",
			"leftrearbumper", 'leftrearbumperclading', "leftorvm", 'leftrearcorner', "frontws", "doorhandle", 'reartowhookcover', 'fronttowhookcover']

			#damage_list = ['BG', 'scratch', 'd1', 'd2', 'd3', 'tear', 'fade', 'bumperdent', 'bumpertear', 'clipsbroken', 'bumpertorn', 'broken', 'shattered', 'cracked']

	#JLR SUV
	elif(make == "Land Rover" and model in ['Range Rover Velar', 'Discovery', 'Discovery Sport']):
		print("Running JLR SUV model.")
		panel_limit = 0.90

		if(weights_type == 'detectron2'):
			panel_list = ['BG', 'bonnet', 'doorhandle', 'frontws', 'leftbootlamp', 'leftfender', 'leftfrontbumper', 'leftfrontdoor', 'leftfrontrocker',
			'leftheadlamp', 'leftorvm', 'leftqpanel', 'leftrearbumper', 'leftreardoor', 'leftrearrocker', 'lefttaillamp', 'rearws', 'rightbootlamp',
			'rightfender', 'rightfrontbumper', 'rightfrontdoor', 'rightfrontrocker', 'rightheadlamp', 'rightorvm', 'rightqpanel', 'rightrearbumper',
			'rightreardoor', 'rightrearrocker', 'righttaillamp', 'tailgate']

			damage_list = ['scratch', 'd2']

		else:
			panel_list = ['BG', 'bonnet', 'doorhandle', 'frontws', 'leftbootlamp', 'leftfender', 'leftfrontbumper', 'leftfrontdoor', 'leftfrontrocker',
			'leftheadlamp', 'leftorvm', 'leftqpanel', 'leftrearbumper', 'leftreardoor', 'leftrearrocker', 'lefttaillamp', 'rearws', 'rightbootlamp',
			'rightfender', 'rightfrontbumper', 'rightfrontdoor', 'rightfrontrocker', 'rightheadlamp', 'rightorvm', 'rightqpanel', 'rightrearbumper',
			'rightreardoor', 'rightrearrocker', 'righttaillamp', 'tailgate']

	#EVOQUE
	elif(make == "Land Rover" and model == "New Range Rover Evoque"):
		print("Running Evoque model.")
		panel_limit = 0.95

		if(weights_type == 'detectron2'):
			panel_list = ["BG", "leftfrontbumper", "rightfrontbumper", "frontbumpercladding", "fronttowhookcover", "rightbumpercam", "leftbumpercam", "rightfrontcorner", "leftfrontcorner", "rearbumper", "rightfoglamp",
			"leftfoglamp", "rightfrontdoor", "leftfrontdoor", "rightfrontrocker", "leftfrontrocker", "rightreardoor", "leftreardoor", "leftrearrocker", "rightrearrocker", "rightqpanel", "leftqpanel",
			"rightfender", "leftfender", "doorhandle", "bonnet", "tailgate", "frontws", "rearws", "rightorvm", "leftorvm", "rightheadlamp", "leftheadlamp", "righttaillamp", "lefttaillamp", "rightrearcladding",
			"leftrearcladding", "rightwa", "leftwa", "rightvalence", "leftvalence", "leftrearwa", "rightrearwa", "rightfrontwa", "leftfrontwa", "rightfenderrocker", "leftfenderrocker"]

			damage_list = ["scratch", "d2"]

		else:
			panel_list = ["BG", "leftfrontbumper", "rightfrontbumper", "frontbumpercladding", "fronttowhookcover", "rightbumpercam", "leftbumpercam", "rightfrontcorner", "leftfrontcorner", "rearbumper", "rightfoglamp",
			"leftfoglamp", "rightfrontdoor", "leftfrontdoor", "rightfrontrocker", "leftfrontrocker", "rightreardoor", "leftreardoor", "leftrearrocker", "rightrearrocker", "rightqpanel", "leftqpanel",
			"rightfender", "leftfender", "doorhandle", "bonnet", "tailgate", "frontws", "rearws", "rightorvm", "leftorvm", "rightheadlamp", "leftheadlamp", "righttaillamp", "lefttaillamp", "rightrearcladding",
			"leftrearcladding", "rightwa", "leftwa", "rightvalence", "leftvalence", "leftrearwa", "rightrearwa", "rightfrontwa", "leftfrontwa", "rightfenderrocker", "leftfenderrocker"]


	#F PACE
	elif(make == "Jaguar" and model in ['F Pace', 'E Pace', 'I Pace']):
		print("Running Jag Pace model.")
		panel_limit = 0.90

		if(weights_type == 'detectron2'):
			panel_list = ['BG', 'bonnet', 'doorhandle', 'frontws', 'leftbootlamp', 'leftfender', 'leftfrontbumper', 'leftfrontdoor',
			'leftfrontrocker', 'leftheadlamp', 'leftorvm', 'leftqpanel', 'leftrearbumper', 'leftreardoor',
			'leftrearrocker', 'lefttaillamp', 'rearws', 'rightbootlamp', 'rightfender', 'rightfrontbumper',
			'rightfrontdoor', 'rightfrontrocker', 'rightheadlamp', 'rightorvm', 'rightqpanel', 'rightrearbumper',
			'rightreardoor', 'rightrearrocker', 'righttaillamp', 'tailgate']
			damage_list = ["scratch", "d2"]

		else:
			panel_list = ['BG', 'bonnet', 'doorhandle', 'frontws', 'leftbootlamp', 'leftfender', 'leftfrontbumper', 'leftfrontdoor',
			'leftfrontrocker', 'leftheadlamp', 'leftorvm', 'leftqpanel', 'leftrearbumper', 'leftreardoor',
			'leftrearrocker', 'lefttaillamp', 'rearws', 'rightbootlamp', 'rightfender', 'rightfrontbumper',
			'rightfrontdoor', 'rightfrontrocker', 'rightheadlamp', 'rightorvm', 'rightqpanel', 'rightrearbumper',
			'rightreardoor', 'rightrearrocker', 'righttaillamp', 'tailgate']

	#JAG SEDAN
	elif(make == "Jaguar" and model in ['XE', 'XF', 'XJ', 'XJL', 'F Type']):
		print("Running Jag Sedan model.")
		panel_limit = 0.90

		if(weights_type == 'detectron2'):
			panel_list = ['BG', 'bonnet', 'doorhandle', 'frontws', 'leftbootlamp', 'leftfender', 'leftfrontbumper', 'leftfrontdoor', 'leftheadlamp',
			'leftorvm', 'leftqpanel', 'leftrearbumper', 'leftreardoor', 'leftrunningboard', 'lefttaillamp', 'rearws', 'rightbootlamp',
			'rightfender', 'rightfrontbumper', 'rightfrontdoor', 'rightheadlamp', 'rightorvm', 'rightqpanel', 'rightrearbumper',
			'rightreardoor', 'rightrunningboard', 'righttaillamp', 'tailgate']

			damage_list = ["scratch", "d2"]

		else: 
			panel_list = ['BG', 'bonnet', 'doorhandle', 'frontws', 'leftbootlamp', 'leftfender', 'leftfrontbumper', 'leftfrontdoor', 'leftheadlamp',
					'leftorvm', 'leftqpanel', 'leftrearbumper', 'leftreardoor', 'leftrunningboard', 'lefttaillamp', 'rearws', 'rightbootlamp',
					'rightfender', 'rightfrontbumper', 'rightfrontdoor', 'rightheadlamp', 'rightorvm', 'rightqpanel', 'rightrearbumper',
					'rightreardoor', 'rightrunningboard', 'righttaillamp', 'tailgate']

	elif (make == "boxvan"):
		panel_list = ['BG', 'bonnet', 'frontbumper', 'leftfender', 'leftfrontdoor', 'leftreardoor', 'leftrunningboard', 'leftqpanel', 'leftorvm', 'leftheadlamp', 'lefttaillamp',
				'leftfoglamp', 'leftwa', 'leftbootlamp', 'rightfender', 'rightfrontdoor', 'rightreardoor', 'rightrunningboard', 'rightqpanel', 'rightorvm', 'rightheadlamp', 'righttaillamp',
				'rightfoglamp', 'rightwa', 'rightbootlamp', 'rearbumper', 'tailgate', 'mtailgate', 'frontws', 'rearws', 'doorhandle']
		damage_list = ['stickertorn']

	else:
			generic_flag = True
			print("Running Generic model.")
			print ("Geneic model name :",model)
			panel_limit = 0.92  # GenericModel_99_108_S87_O88_P92_genericData_19082019
			if(weights_type == 'detectron2'):
				panel_list = ['alloywheel', 'tyre', 'rightfoglamp', 'rightheadlamp', 'frontbumper', 'licenseplate', 'bonnet',
				'leftheadlamp', 'lowerbumpergrille', 'tailgate', 'logo', 'rightrunningboard', 'lefttaillamp',
				'fuelcap', 'rearbumper', 'namebadge', 'leftorvm', 'rightorvmbroken', 'rightfrontdoor', 'doorhandle',
				'rightfrontdoorcladding', 'rightreardoorcladding', 'Reflector', 'rightwa', 'headlightwashermissing', 'towbarcover',
				'rightfoglampmissing', 'sensormissing', 'rightbootlamp', 'righttaillamp', 'mtailgate', 'rightqpanel', 'frontbumpergrille',
				'leftbootlamp', 'rearws', 'leftqpanel', 'leftwa', 'leftorvmbroken', 'leftfrontventglass', 'sensor', 'headlightwasher',
				'rightapillar', 'leftrunningboard', 'alloy curbrash', 'rightrearventglass', 'rightreardoorglass', 'rightdpillar',
				'leftfender', 'leftheadlampmissing', 'wheelcap', 'leftapillar', 'leftfrontdoorglass', 'leftfoglampmissing', 'rightreardoor',
				'variant', 'leftfoglamp', 'rightfender', 'leftfrontdoor', 'rightfrontdoorglass', 'frontbumpergrillemissing', 'leftcpillar',
				'leftrearventglass', 'leftreardoorglass', 'leftreardoor', 'rightorvm', 'rightbpillar','frontws','footstep','leftfrontdoorcladding','leftreardoorcladding',
				'leftbootlamp', 'alloywheel', 'rearbumper', 'leftqpanel', 'leftreardoorglass', 'Reflector', 'rightfrontdoorglass', 'tyre', 'rightbootlamp', 'leftroofside', 
    			'leftfrontdoorcladding', 'rightreardoorglass', 'rightheadlamp', 'logo', 'leftrunningboard', 'leftfrontdoor', 'rightfender', 'frontbumper', 'rightqpanel',
       			'frontws', 'towbarcover', 'rightfoglamp', 'indicator', 'rightorvm', 'rightfrontdoor', 'rightorvmbroken', 'leftfrontdoorglass', 'rightfrontdoorcladding', 
          		'rightrearventglass', 'leftapillar', 'tailgate', 'leftreardoorcladding', 'wheelcap curbrash', 'rightcpillar', 'doorhandle', 'rightapillar', 'leftorvm', 
            	'wheelrim', 'leftwa', 'mtailgate', 'wheelcap', 'sensor', 'licenseplate', 'roofrail', 'bonnet', 'frontbumpercladding', 'variant', 'leftfender', 'rearws',
             	'rightreardoor', 'lefttaillamp', 'lowerbumpergrille', 'leftbpillar', 'alloy curbrash', 'frontbumpergrille', 'footstep', 'fuelcap', 'rightreardoorcladding',
              	'rightroofside', 'righttaillamp', 'namebadge', 'leftreardoor', 'leftcpillar', 'rightrunningboard', 'rightbpillar', 'leftrearventglass', 'leftfoglamp',
            	'rearbumpercladding', 'rightwa', 'leftheadlamp', 'wiper']

				damage_list = ['scratch', 'd2', 'tear', 'clipsbroken',  'shattered', 'broken']

			else:
				panel_list = ['BG', 'bonnet', 'frontbumper', 'leftfender', 'leftfrontdoor', 'leftreardoor', 'leftrunningboard', 'leftqpanel', 'leftorvm', 'leftheadlamp', 'lefttaillamp',
				'leftfoglamp', 'leftwa', 'leftbootlamp', 'rightfender', 'rightfrontdoor', 'rightreardoor', 'rightrunningboard', 'rightqpanel', 'rightorvm', 'rightheadlamp', 'righttaillamp',
				'rightfoglamp', 'rightwa', 'rightbootlamp', 'rearbumper', 'tailgate', 'mtailgate', 'frontws', 'rearws', 'doorhandle']


	class_names = list(set(panel_list)) + damage_list # concat
	print('make:',make,flush=True)
	print('model',model,flush=True)
	print('Damage Weights:',weights_type,flush=True)
	print('class:',class_names,flush=True)
	print('damage list:',damage_list,flush=True)

	return list(set(panel_list)),damage_list
