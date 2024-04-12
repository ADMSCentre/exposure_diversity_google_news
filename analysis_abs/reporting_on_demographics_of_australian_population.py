'''
	
	ABS Census Interpretation Notes:

	* We first refer to the file '<YEAR>Census_geog_desc_1st_2nd_3rd_release.xlsx', which can be found at any path
	of the following form:

		data/census/<ANY_STATE>/Metadata/<YEAR>Census_geog_desc_1st_2nd_3rd_release.xlsx

	Within this file, we note that 'STE' codes correspond directly to statistics gathered for entire states, as noted
	as the largest spatial unit in the ASGS and are part of the Main Structure within ABS Structures.

		Refer to https://www.abs.gov.au/ausstats/abs@.nsf/Lookup/2901.0Chapter23002016#STE for more details

	Thereafter, we can isolate any folder of the following path form to select any statistic (DESIRED_CODE) by any state (ANY_STATE)

		data/census/<ANY_STATE>/data/STE/<ANY_STATE>/<DESIRED_CODE>

	The analysis for each relevant code is given as follows

	General note: There is a caveat of interpolating values for age ranges, which is described as follows:

		Consider that a cohort of 721 15 - 24 year olds are adjacently indicated next to a cohort of 13,999 25 - 34 year olds.
		Our interpolation function attempts to map the cohort to a new age range of 18 - 24 year olds, and obtains a simulated cohort
		of 2712 participants. This is a behaviour of the interpolation, despite the original age range and the target age range being aligned at
		the upper bound.

'''

import os
import sys
import ipdb
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

'''
	This function accepts a 1-D array of frequencies for a range (A), a second range (B), and returns the values 
	for the frequencies at B, given that we assume the frequencies of range A also form a spline.
'''
def spline_projection_from_frequencies(A_frequencies_raw, A_range_raw, B_range):
	return [np.interp(b, A_range_raw, A_frequencies_raw) for b in B_range]

'''
	This function returns the midpoint of two values
'''
def midpoint(min_x, max_x):
	return ((min_x+max_x)/2)

THIS_YEAR = "2021"
ASGS_STATES = ["QLD", "NSW", "VIC", "ACT", "NT", "WA", "TAS", "SA"]
ASE_BACKMAP_GENDERS = {"M": "Male", "F": "Female"}
MAXIMUM_AGE_BOUNDARY = 200

ASE_CATEGORY_LABELS = {
	"gender" : ["Male", "Female", "Undescribed"],
	"state" : ["NSW", "VIC", "QLD", "SA", "WA", "TAS", "NT", "ACT", "Undescribed"],
	"employment" : ["Employed", "Unemployed", "Retired", "Undescribed"],#"Employed full-time", "Employed part-time", "Unemployed and looking for work", "Unemployed and not looking for work", "Retired", "Undescribed"],
	"age" : ["18 - 34", "35 - 64", "65 and over", "Undescribed"],#"18 - 24", "25 - 34", "35 - 44", "45 - 54", "55 - 64", "65 - 74", "75 and over", "Undescribed"],
	"education" : ["Secondary or less", "Tertiary", "Undescribed"],#"Year 12 or equivalent", "Less than year 12 or equivalent", "Postgraduate degree level", "Bachelor degree level", "Undescribed"],
	"party" : ["Australian Labor Party", "Liberal National Party", "Australian Greens", "Undescribed"],#"One Nation", "Undescribed"],
	"income" : ["$1 - $51,999", "$52,000 - $103,999", "$104,000 - $155,999", "$156,000 or more", "Undescribed"]#["$1 - $15,599", "$15,600 - $20,799", "$20,800 - $25,999", "$26,000 - $33,799", "$33,800 - $41,599", "$41,600 - $51,999", "$52,000 - $64,999", "$65,000 - $77,999", "$78,000 - $90,999", "$91,000 - $103,999", "$104,000 - $155,999", "$156,000 or more", "Undescribed"]
}

ELECTORAL_COMMISSION_CODES = {
	"LP" : "Liberal National Party",
	"LNP" : "Liberal National Party",
	"NP" : "Liberal National Party",
	"ALP" : "Australian Labor Party",
	"GRN" : "Australian Greens",
	"Undescribed" : "Undescribed"
}

ASE_BACKMAP_AGE = {
		"18 - 24" : list(range(18,24+1)),
		"25 - 34" : list(range(25,34+1)),
		"35 - 44" : list(range(35,44+1)),
		"45 - 54" : list(range(45,54+1)),
		"55 - 64" : list(range(55,64+1)),
		"65 - 74" : list(range(65,74+1)),
		"75 and over" : list(range(75,MAXIMUM_AGE_BOUNDARY+1)),
	}

ASE_BACKMAP_AGE = {
		"18 - 34" : list(range(18,34+1)),
		"35 - 64" : list(range(35,64+1)),
		"65 and over" : list(range(65,MAXIMUM_AGE_BOUNDARY+1)),
	}

ASE_BACKMAP_INCOME = {
		"$1 - $15,599" : list(range(1,15599+1)),
		"$15,600 - $20,799" : list(range(15600,20799+1)),
		"$20,800 - $25,999" : list(range(20800,25999+1)),
		"$26,000 - $33,799" : list(range(26000,33799+1)),
		"$33,800 - $41,599" : list(range(33800,41599+1)),
		"$41,600 - $51,999" : list(range(41600,51999+1)),
		"$52,000 - $64,999" : list(range(52000,64999+1)),
		"$65,000 - $77,999" : list(range(65000,77999+1)),
		"$78,000 - $90,999" : list(range(78000,90999+1)),
		"$91,000 - $103,999" : list(range(91000,103999+1)),
		"$104,000 - $155,999" : list(range(104000,155999+1))
	}

ASE_BACKMAP_INCOME = {
		"$1 - $51,999" : list(range(1,51999+1)),
		"$52,000 - $103,999" : list(range(52000,103999+1)),
		"$104,000 - $155,999" : list(range(104000,155999+1))
	}

'''
	Proceeding this function, the majority of data we retrieve from the ABS Census follows a common method of extraction. This method
	involves distributing the requested data by state, and converting the data (which is given in CSV format) to JSON format.

	Here the functionality is implemented, given a filename part (CENSUS_FILENAME_PART), as well as other details
'''
def abs_csv_to_dict(
		CENSUS_FILENAME_PART,
		OMITTED_KEYS=[], 
		THIS_STATE="<THIS_STATE>"):
	# The indicators associated with census code G04
	SOURCE_FILENAME_TEMPLATE = os.path.join("data","census",THIS_STATE,"data","STE",THIS_STATE, CENSUS_FILENAME_PART + f"<THIS_ALPHABET_INDICATOR>_{THIS_STATE}_STE.csv")

	alphabetical_indicators = [str()] + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

	# Source filenames
	source_files = list()
	# All data from census by state
	data_by_state_dict = dict()
	# For each of the states
	for state in ASGS_STATES:
		# This dictionary will be populated by two files, which correspond to the alphabet indicators
		data_dict = dict()
		# Assistant dictionary
		data_dict_assist = list()
		# For each of the indicators
		for alphabet_indicator in alphabetical_indicators:
			# Add the necessary file path, as localised to state and indicator
			source_fname = (SOURCE_FILENAME_TEMPLATE.replace(THIS_STATE, state)
														.replace("<THIS_ALPHABET_INDICATOR>", alphabet_indicator))
			this_attempted_source_fname = os.path.join(os.getcwd(), source_fname)
			if os.path.exists(this_attempted_source_fname):
				# Open the source file
				source_file_data = open(this_attempted_source_fname).read().split("\n")
				source_files.append(source_fname)
				data_dict_assist = list()
				# For the source file's data
				for i in range(len(source_file_data)):
					# Split each line's data by the delimiter
					source_file_data_line_values = source_file_data[i].split(",")
					if (i == 0):
						# First line is titles, which corresponds to keys
						for k in source_file_data_line_values:
							data_dict[k] = None
							data_dict_assist.append(k)
					elif (i == 1):
						# Second line is values, which must be mapped back to keys
						for j in range(len(source_file_data_line_values)):
							data_dict[data_dict_assist[j]] = float(source_file_data_line_values[j])
			else:
				pass
		# Apply the omissions
		for k in OMITTED_KEYS:
			del data_dict[k]
		'''
		# Print the data for a single state
		print(json.dumps(data_dict,indent=3))
		'''
		data_by_state_dict[state] = data_dict
	'''
	# Print the source files for the analysis
	print(json.dumps(source_files,indent=3))
	'''
	return data_by_state_dict

'''
	Age-Gender-State Correlation 

	* We refer to the file 'Metadata_<YEAR>_GCP_DataPack.xlsx', which can be found at any path
	of the following form:
		
		data/census/<ANY_STATE>/Metadata/Metadata_<YEAR>_GCP_DataPack.xlsx

	Herein, we find that the code 'G04' corresponds to 'Age by Sex' of relevant persons.

	Then, we refer to the following paths for ABS Census 'Age by Sex' statistics:

		data/census/<ANY_STATE>/data/STE/<ANY_STATE>/<YEAR>Census_G04A_<ANY_STATE>_STE.csv
		data/census/<ANY_STATE>/data/STE/<ANY_STATE>/<YEAR>Census_G04B_<ANY_STATE>_STE.csv

	Separating these paths by state, we can determine 'Age-Gender-State' correlation

'''

'''
	This function returns the total number of individuals surveyed by the ABS Census,
	by state, age range, and gender, in the language of the categories defined in the ASE.
'''
def interpret_abs_G04():
	# Retrieve the raw data from the ABS Census
	state_age_gender_dict = abs_csv_to_dict(THIS_YEAR+"Census_G04", OMITTED_KEYS=["STE_CODE_"+THIS_YEAR])
	'''
	# Print the state-age-gender dictionary
	print(json.dumps(state_age_gender_dict,indent=3))
	'''
	'''
		In the next step, we process the data taken from the census, to label it according to the fields
		of the ASE.

		Note: We make a decision here to disinclude the 'Prefer not to say' field for all categories, since
		the census does not offer this option to its surveyed participants for this topic.
	'''
	ase_state_age_gender_dict = dict() 
	for this_state in state_age_gender_dict.keys():
		for field in state_age_gender_dict[this_state].keys():
			# We run an exclusion on total fields, given our decoder script works only with specific age/gender ranges
			if ("Tot" not in field):
				# We implement some string functions to improve processing, and help our decoder
				field_processed = field.replace("Age_yr_","").replace("_yr_over_","_%s_" % (MAXIMUM_AGE_BOUNDARY))
				# Inhibiting the 'total population' value, as it is not necessary
				if (field_processed[-1:] != "P"):
					this_gender = ASE_BACKMAP_GENDERS[field_processed[-1:]]
					# The age range is isolated from the rest of the field's title and split by the underscore delimiter
					this_age_range = [int(x) for x in field_processed[:-2].split("_")]
					# Only quantify ranges for ages equal to or above 80, as the ages for these ranges do not have itemised versions
					if any([True if (x >= 80) else False for x in this_age_range]):
						'''
							The lower and upper bound are calculated (modulo operator guards against instances where the 
							range's upper and lower bound are the same number, although it isn't expected that this occurs for ages 80+
							since they are not itemised
						'''
						age_bound_lower = this_age_range[0%len(this_age_range)]
						age_bound_upper = this_age_range[1%len(this_age_range)]
						# The new age range is declared
						this_age_range = list(range(age_bound_lower,age_bound_upper+1))
					else:
						# If we are dealing with a single age
						if (len(this_age_range) == 1):
							# Allow it to be processed
							this_age_range = list(range(this_age_range[0],this_age_range[0]+1))
						else:
							'''
								Or else do not allow it to be processed, since age ranges of more than a single age that are less than 80
								overlap single ages that are also indexed.
							'''
							this_age_range = None

					# If the age range can be processed
					if (this_age_range is not None):
						# Refer the age range to backmap dictionary
						for k,v in ASE_BACKMAP_AGE.items():
							# The key from the backmap is derived from its value's intersection with the given age range.
							if (len(set(v).intersection(set(this_age_range))) > 0):
								this_age_range = k
								break
						if (type(this_age_range) is not list):
							# Append the value for all three linked results to the ASE list
							if (this_state not in ase_state_age_gender_dict.keys()):
								ase_state_age_gender_dict[this_state] = dict()
							if (this_age_range not in ase_state_age_gender_dict[this_state].keys()):
								ase_state_age_gender_dict[this_state][this_age_range] = dict()
							if (this_gender not in ase_state_age_gender_dict[this_state][this_age_range].keys()):
								ase_state_age_gender_dict[this_state][this_age_range][this_gender] = 0
							ase_state_age_gender_dict[this_state][this_age_range][this_gender] += state_age_gender_dict[this_state][field]
	'''
	# Print the distribution of individuals surveyed by the census, by state, age, and gender
	print(json.dumps(ase_state_age_gender_dict,indent=3))
	'''
	return ase_state_age_gender_dict

'''
	Age-Gender-State-Employment Correlation 

	* We refer to the file 'Metadata_<YEAR>_GCP_DataPack.xlsx', which can be found at any path
	of the following form:
		
		data/census/<ANY_STATE>/Metadata/Metadata_<YEAR>_GCP_DataPack.xlsx

	Herein, we find that the code 'G43' corresponds to 'Labour Force Status by Age by Sex' of relevant persons.

	Then, we refer to the following paths for ABS Census 'Age by Sex' statistics:

		data/census/<ANY_STATE>/data/STE/<ANY_STATE>/<YEAR>Census_G43A_<ANY_STATE>_STE.csv
		data/census/<ANY_STATE>/data/STE/<ANY_STATE>/<YEAR>Census_G43B_<ANY_STATE>_STE.csv

	Separating these paths by state, we can determine 'Age-Gender-State-Employment' correlation

	* Incorporating the 'Retired' category:

		This is acheived by integrating separate statistics that correspond to the number of retired individuals recorded
		for all states, genders, and age ranges (aggregating individuals below 30 years of age) in Australia. This data is taken
		from the following source:

			* Retirement and Retirement Intentions, Australia : 2016 - 2017 financial year & 2018 - 2019 financial year
				data/census2019/62380do013_2016201706.xls
				data/census2019/Graph 1 - Age distribution of retirees aged 45 years and over.csv

			These sources have synthesized the following data file:

				data/census2019/retirees.json


		Integration these statistics has required that the population sizes recorded are consistent with those surveyed in the
		ABS Census. To verify this, we refer to the table 'Australian Bureau of Statistics - Retirement by state and territory',
		where a sample of 1,100,000 individuals aged 45 and over retired in NSW in 2016-2017 financial year. During the same year,
		our ABS Census records state that 1,446,721 individuals were in the workforce. Combining both figures, we determine
		that the retirees make up ~43.1% of the population. Allowing for a 6% error margin, this aligns with the ABS statistic that the
		retirees account for ~36.8% of the population.

'''

'''
	This function returns the total number of individuals surveyed by the ABS Census,
	by state, age range, gender, and employment status, in the language of the categories defined in the ASE.
'''
def interpret_abs_G43_or_G46():
	'''
		These fields are known from the 'Cell descriptors information', at 'Metadata_<YEAR>_GCP_DataPack.xlsx',
		and map to the ASE employment types, as given below.

		Note: The 'Tot_Unemp' will have the 'Unemployed - looking for work' fields subtracted to become the 'Unemployed and not 
		looking for work' field.
	'''
	ASE_BACKMAP_EMPLOYMENT = {
			"Emp_FullT" : "Employed",#"Employed full-time",
			"Emplyed_wrked_full_time" : "Employed",
			"Emp_PartT" : "Employed",#"Employed part-time",
			"Emplyed_wrked_part_time" : "Employed",
			"Tot_Unemp" : "Unemployed",# and not looking for work",
			"Emp_awy_f_wrk" : "Employed",#"Other",
			"Employed_away_from_work" : "Employed",
			"Unmplyed_lookng_for_wrk" : "Unemployed",
			"Hours_wkd_NS" : "Undescribed"#"Prefer not to say"
		}
	table_no = "G43" if (THIS_YEAR == "2016") else "G46" # Between 2016 and 2021, the ABS has changed distributions of employment by age and sex
	state_age_gender_employment_dict = abs_csv_to_dict(THIS_YEAR+"Census_"+table_no, OMITTED_KEYS=["STE_CODE_"+THIS_YEAR])
	'''
	# Print the state-age-gender-employment dictionary
	print(json.dumps(state_age_gender_employment_dict,indent=3))
	'''
	#sys.exit()
	'''
		In the next step, we process the data taken from the census, to label it according to the fields
		of the ASE.
	'''
	ase_state_age_gender_employment_dict = dict() 
	# For each state
	for this_state,data in state_age_gender_employment_dict.items():	
		for field, value in data.items():
			# Set the gender from the ASE backmap (excluding unaccounted fields)
			if (field[0] in ASE_BACKMAP_GENDERS):
				this_gender = ASE_BACKMAP_GENDERS[field[0]]
				# If we are not dealing with a 'total' field
				if (not field.endswith("_Tot")):
					this_employment = None
					for employment_code, ase_employment_code in ASE_BACKMAP_EMPLOYMENT.items():
						if (employment_code in field):
							this_employment = ase_employment_code
							break
					# For 'total' type employments, the 'this_employment' field will be 'None'
					if (this_employment is not None):
						'''
							In this step, we adjust the range of the '85 and over' group to '85'. The reason being
							that all other relevant ages are communicated in two-part ranges for this data. Although we
							are making an assumption that 85 and over only constitutes 85 year olds, this assumption will
							be nullified when we map the results to the ASE age ranges, which aggregate any results for ages beyond
							74 years old.
						'''
						adjusted_field = field.replace("85ov", "85_85")
						# Determine the age range of the field
						this_age_range = [int(x) for x in adjusted_field.split("_")[-2:]]
						this_age_range_midpoint = str(int(midpoint(this_age_range[0], this_age_range[1])))
						# Append the value for all four linked results to the ASE list
						if (this_state not in ase_state_age_gender_employment_dict.keys()):
							ase_state_age_gender_employment_dict[this_state] = dict()
						if (this_gender not in ase_state_age_gender_employment_dict[this_state].keys()):
							ase_state_age_gender_employment_dict[this_state][this_gender] = dict()
						if (this_employment not in ase_state_age_gender_employment_dict[this_state][this_gender].keys()):
							ase_state_age_gender_employment_dict[this_state][this_gender][this_employment] = dict()
						'''
							Note that age ranges are only given as midpoints here - this is required for the next step, that maps the values to the
							ASE ranges
						'''
						if (this_age_range_midpoint
								not in ase_state_age_gender_employment_dict[this_state][this_gender][this_employment].keys()):
							ase_state_age_gender_employment_dict[this_state][this_gender][this_employment][this_age_range_midpoint] = 0
						ase_state_age_gender_employment_dict[this_state][this_gender][this_employment][this_age_range_midpoint] += int(value)
	'''
		In this step, we add the retirement details for each state, gender, and age range
	'''
	retirees_data = json.loads(open(os.path.join("data","census2019","retirees.json")).read())
	# We need to adjust the age ranges of the age ratios, as taken from the ABS 2019 Census
	for this_gender in retirees_data["age_ratios_of_retirement"].keys():
		# Get the dataset, and create a copy
		census_age_midpoint_values_dataset = retirees_data["age_ratios_of_retirement"][this_gender].copy()
		# Formalise the age midpoints of the census...
		age_midpoints = [midpoint(int(x.split("_")[0]), int(x.split("_")[1])) for x in census_age_midpoint_values_dataset.keys()]
		# ...and formalise the associated values as well...
		age_values = [x for x in census_age_midpoint_values_dataset.values()]
		# ASE age midpoints
		ase_age_midpoints = [midpoint(x[0], x[-1]) for x in ASE_BACKMAP_AGE.values()]
		# Then we calculate the values, as they would've been for the ASE
		new_values = spline_projection_from_frequencies(age_values, age_midpoints, ase_age_midpoints)
		new_values_total = sum(new_values)
		new_values = [x/new_values_total for x in new_values]
		# Finally, we reinsert the values into the original dictionary
		retirees_data["age_ratios_of_retirement"][this_gender] = { 
								list(ASE_BACKMAP_AGE.keys())[i] : new_values[i] for i in range(len(ASE_BACKMAP_AGE)) }
	# Then we combine the age range ratios with the state-gender population sizes
	for this_state in retirees_data["states"].keys():
		for this_gender in retirees_data["states"][this_state].keys():
			# The ABS 2019 Census normalises the total values as per 1000
			total_population = retirees_data["states"][this_state][this_gender]*1000
			retirees_data["states"][this_state][this_gender] = {k: v*total_population for k,v in retirees_data["age_ratios_of_retirement"][this_gender].items()}
	'''	
		In this step, we take each dataset, relative to 'state', 'gender', and 'employment', and we map the distribution of values 
		collected for the ages recorded in the ABS Census, to the age ranges of ASE. This is achieved by finding the interpolation of
		Census values at the midpoint of each ASE age range, as is calculated below:
	'''
	for this_state in ase_state_age_gender_employment_dict.keys():
		for this_gender in ase_state_age_gender_employment_dict[this_state].keys():
			for this_employment in ase_state_age_gender_employment_dict[this_state][this_gender].keys():
				# Get the dataset, and create a copy
				census_age_midpoint_values_dataset = ase_state_age_gender_employment_dict[this_state][this_gender][this_employment].copy()
				# Formalise the age midpoints of the census...
				age_midpoints = [int(x) for x in census_age_midpoint_values_dataset.keys()]
				# ...and formalise the associated values as well...
				age_values = [x for x in census_age_midpoint_values_dataset.values()]
				# ...and also formalise the ASE age midpoints
				ase_age_midpoints = [midpoint(x[0], x[-1]) for x in ASE_BACKMAP_AGE.values()]
				'''
					We deliberately set the last midpoint of the ASE age midpoints to only the first index of its range, to avoid any interpolation 
					cutoff issues
				'''
				ase_age_midpoints[-1] = list(ASE_BACKMAP_AGE.values())[-1][0]
				# Then we calculate the values, as they would've been for the ASE
				new_values = spline_projection_from_frequencies(age_values, age_midpoints, ase_age_midpoints)
				#print(new_values)
				#sys.exit()
				# Finally, we reinsert the values into the original dictionary
				ase_state_age_gender_employment_dict[this_state][this_gender][this_employment] = { 
														list(ASE_BACKMAP_AGE.keys())[i] : new_values[i] for i in range(len(ASE_BACKMAP_AGE)) }
			ase_state_age_gender_employment_dict[this_state][this_gender]["Retired"] = retirees_data["states"][this_state][this_gender]
	'''
		Execute the necessary subtraction to derive the 'Unemployed and not looking for work' field
	'''
	'''
	for this_state in ase_state_age_gender_employment_dict.keys():
		for this_gender in ase_state_age_gender_employment_dict[this_state].keys():
			for this_age_range in ase_state_age_gender_employment_dict[this_state][this_gender]["Unemployed and not looking for work"].keys():
				ase_state_age_gender_employment_dict[this_state][this_gender]["Unemployed and not looking for work"][this_age_range] -= \
					ase_state_age_gender_employment_dict[this_state][this_gender]["Unemployed and looking for work"][this_age_range]
				# Interpolating will result in negative values - we overcome this by absolute values
				ase_state_age_gender_employment_dict[this_state][this_gender]["Unemployed and not looking for work"][this_age_range] = \
					abs(ase_state_age_gender_employment_dict[this_state][this_gender]["Unemployed and not looking for work"][this_age_range])
	'''
	'''
	# Print the distribution of individuals surveyed by the census, by state, age, gender, and employment
	print(json.dumps(ase_state_age_gender_employment_dict,indent=3))
	'''
	return ase_state_age_gender_employment_dict

'''
	Age-Gender-State-Income Correlation 

	* We refer to the file 'Metadata_<YEAR>_GCP_DataPack.xlsx', which can be found at any path
	of the following form:
		
		data/census/<ANY_STATE>/Metadata/Metadata_<YEAR>_GCP_DataPack.xlsx

	Herein, we find that the code 'G17' corresponds to 'Total Personal Income (Weekly) by Age by Sex' of relevant persons.

	Then, we refer to the following paths for ABS Census 'Total Personal Income (Weekly) by Age by Sex' statistics:

		data/census/<ANY_STATE>/data/STE/<ANY_STATE>/<YEAR>Census_G17A_<ANY_STATE>_STE.csv
		data/census/<ANY_STATE>/data/STE/<ANY_STATE>/<YEAR>Census_G17B_<ANY_STATE>_STE.csv
		data/census/<ANY_STATE>/data/STE/<ANY_STATE>/<YEAR>Census_G17C_<ANY_STATE>_STE.csv

	Separating these paths by state, we can determine 'Age-Gender-State-Income' correlation
'''

'''
	This function returns the total number of individuals surveyed by the ABS Census,
	by state, age range, gender, and income, in the language of the categories defined in the ASE.
'''
def interpret_abs_G17():
	state_age_gender_income_dict = abs_csv_to_dict(THIS_YEAR+"Census_G17", OMITTED_KEYS=["STE_CODE_"+THIS_YEAR])
	state_age_gender_income_dict_assist = dict()
	for this_state in state_age_gender_income_dict.keys():
		for field, value in state_age_gender_income_dict[this_state].items():
			# Set the gender from the ASE backmap (excluding unaccounted fields)
			if (field[0] in ASE_BACKMAP_GENDERS):
				this_gender = ASE_BACKMAP_GENDERS[field[0]]
				# Do not process the field if it is a total field
				if (not "_Tot" in field):
					# Generate new ranges for boundary income ranges
					adjusted_field = field.replace("3000_more", "3000_3000")
					for nil_value in ["Neg_Nil_income", "Negtve_Nil_incme", "Neg_Nil_incme"]:
						adjusted_field = adjusted_field.replace(nil_value, "0_0")
					# Generate new ranges for boundary age ranges
					adjusted_field = adjusted_field.replace("85_yrs_ovr","85_85_yrs").replace("85ov", "85_85_yrs")
					# Remove the trailing 'years' suffix
					adjusted_field = adjusted_field.replace("_yrs","")
					# Assume the income range midpoint is this value to begin with...
					this_income_range_midpoint = "Undescribed"
					if (not "PI_NS" in field):
						# Calculate ranges and midpoints for incomes and ages
						this_income_range = [int(x) for x in adjusted_field.split("_")[1:2+1] if (not x in ["more"])]
						'''
							The income range is multiplied by 52 (as in 52 weeks), given that the income is described according to a 'weekly' interval.
							Thereafter, it is rounded, and converted into a string, so that it may be referenced as a key.
						'''
						if (len(this_income_range) > 1):
							this_income_range_midpoint = str(int(round(midpoint(this_income_range[0], this_income_range[1])*52)))
						else:
							this_income_range_midpoint = str(int(round(this_income_range[0]*52)))
					this_age_range = [int(x) for x in [adjusted_field.split("_")[-2],adjusted_field.split("_")[-1]]]
					this_age_range_midpoint = midpoint(this_age_range[0], this_age_range[1])
					'''
						In this step, we assemble the 'state_age_gender_income_dict_assist' dictionary, which only communicates the midpoints
						of the income and age ranges of the ABS Census. In the proceeding steps, we will map these fields into those of the ASE.
					'''
					if (this_state not in state_age_gender_income_dict_assist.keys()):
						state_age_gender_income_dict_assist[this_state] = dict()
					if (this_gender not in state_age_gender_income_dict_assist[this_state].keys()):
						state_age_gender_income_dict_assist[this_state][this_gender] = dict()
					if (this_income_range_midpoint not in state_age_gender_income_dict_assist[this_state][this_gender].keys()):
						state_age_gender_income_dict_assist[this_state][this_gender][this_income_range_midpoint] = dict()
					if (this_age_range_midpoint not in state_age_gender_income_dict_assist[this_state][this_gender][this_income_range_midpoint].keys()):
						state_age_gender_income_dict_assist[this_state][this_gender][this_income_range_midpoint][this_age_range_midpoint] = 0
					state_age_gender_income_dict_assist[this_state][this_gender][this_income_range_midpoint][this_age_range_midpoint] += int(value)
	'''
		In this step, we take each dataset, relative to 'state', 'gender', and 'ABS income midpoint', and we map the distribution of values 
		collected for the ages recorded in the ABS Census, to the age ranges of ASE. This is achieved by finding the interpolation of
		Census values at the midpoint of each ASE age range, as is calculated below:
	'''
	for this_state in state_age_gender_income_dict_assist.keys():
		for this_gender in state_age_gender_income_dict_assist[this_state].keys():
			for this_income_range_midpoint in state_age_gender_income_dict_assist[this_state][this_gender].keys():
				# Get the dataset, and create a copy
				census_age_midpoint_values_dataset = state_age_gender_income_dict_assist[this_state][this_gender][this_income_range_midpoint].copy()
				# Formalise the age midpoints of the census...
				age_midpoints = [int(x) for x in census_age_midpoint_values_dataset.keys()]
				# ...and formalise the associated values as well...
				age_values = [x for x in census_age_midpoint_values_dataset.values()]
				# ...and also formalise the ASE age midpoints
				ase_age_midpoints = [midpoint(x[0], x[-1]) for x in ASE_BACKMAP_AGE.values()]
				'''
					We deliberately set the last midpoint of the ASE age midpoints to only the first index of its range, to avoid any interpolation 
					cutoff issues
				'''
				ase_age_midpoints[-1] = list(ASE_BACKMAP_AGE.values())[-1][0]
				# Then we calculate the values, as they would've been for the ASE
				new_values = spline_projection_from_frequencies(age_values, age_midpoints, ase_age_midpoints)
				# Finally, we reinsert the values into the original dictionary
				state_age_gender_income_dict_assist[this_state][this_gender][this_income_range_midpoint] = { 
														list(ASE_BACKMAP_AGE.keys())[i] : new_values[i] for i in range(len(ASE_BACKMAP_AGE)) }
	'''
		Print the assistant dictionary 'state_age_gender_income_dict_assist', after the age midpoints from the ABS Census
		are mapped to the ASE age ranges
	'''
	#print(json.dumps(state_age_gender_income_dict_assist,indent=3))
	'''
		In the next step, we have to 'fold the assistant dictionary upon itself' to group values by ABS Census income midpoints
	'''
	# We need to declare a second assistant dictionary for the regrouped values
	state_age_gender_income_dict_assist_2 = dict()
	for this_state in state_age_gender_income_dict_assist.keys():
		for this_gender in state_age_gender_income_dict_assist[this_state].keys():
			for this_income_range_midpoint in state_age_gender_income_dict_assist[this_state][this_gender].keys():
				for this_age_range, value in state_age_gender_income_dict_assist[this_state][this_gender][this_income_range_midpoint].items():
					if (this_state not in state_age_gender_income_dict_assist_2.keys()):
						state_age_gender_income_dict_assist_2[this_state] = dict()
					if (this_gender not in state_age_gender_income_dict_assist_2[this_state].keys()):
						state_age_gender_income_dict_assist_2[this_state][this_gender] = dict()
					if (this_age_range not in state_age_gender_income_dict_assist_2[this_state][this_gender].keys()):
						state_age_gender_income_dict_assist_2[this_state][this_gender][this_age_range] = dict()
					if (this_income_range_midpoint not in state_age_gender_income_dict_assist_2[this_state][this_gender][this_age_range].keys()):
						state_age_gender_income_dict_assist_2[this_state][this_gender][this_age_range][this_income_range_midpoint] = value
	'''
		Print the second assistant dictionary 'state_age_gender_income_dict_assist_2', after it has been regrouped for ABS Census income midpoints
	'''
	#print(json.dumps(state_age_gender_income_dict_assist_2,indent=3))
	'''
		Like with the age ranges, in this step, we take each dataset, relative to 'state', 'gender', and 'age', and we map the distribution of 
		values collected for the incomes recorded in the ABS Census, to the income ranges of ASE. This is achieved by finding the 
		interpolation of Census values at the midpoint of each ASE income range, as is calculated below:
	'''
	for this_state in state_age_gender_income_dict_assist_2.keys():
		for this_gender in state_age_gender_income_dict_assist_2[this_state].keys():
			for this_age_range in state_age_gender_income_dict_assist_2[this_state][this_gender].keys():
				# Get the dataset, and create a copy
				census_income_midpoint_values_dataset = state_age_gender_income_dict_assist_2[this_state][this_gender][this_age_range].copy()
				# Formalise the income midpoints of the census...
				income_midpoints = [int(x) for x in census_income_midpoint_values_dataset.keys() if (x not in ["Undescribed"])]
				# ...and formalise the associated values as well...
				income_values = [v for k,v in census_income_midpoint_values_dataset.items() if (k not in ["Undescribed"])]
				# ...and also formalise the ASE income midpoints
				ase_income_midpoints = [midpoint(x[0], x[-1]) for x in ASE_BACKMAP_INCOME.values()]
				# Then we calculate the values, as they would've been for the ASE
				new_values = spline_projection_from_frequencies(income_values, income_midpoints, ase_income_midpoints)
				# Finally, we reinsert the values into the original dictionary
				state_age_gender_income_dict_assist_2[this_state][this_gender][this_age_range] = { 
											list(ASE_BACKMAP_INCOME.keys())[i] : new_values[i] for i in range(len(ASE_BACKMAP_INCOME)) }
				state_age_gender_income_dict_assist_2[this_state][this_gender][this_age_range]["Undescribed"] = census_income_midpoint_values_dataset["Undescribed"]
	'''
	# Print the distribution of individuals surveyed by the census, by state, age, gender, and income
	print(json.dumps(state_age_gender_income_dict_assist_2,indent=3))
	'''
	return state_age_gender_income_dict_assist_2

'''
	Age-Gender-State-Education Correlation 

	* We refer to the file 'Metadata_<YEAR>_GCP_DataPack.xlsx', which can be found at any path
	of the following form:
		
		data/census/<ANY_STATE>/Metadata/Metadata_<YEAR>_GCP_DataPack.xlsx

	Herein, we find that the code 'G16' and 'G43' correspond to 'Highest Year of School Completed by Age by Sex' and 
	'Non-School Qualification:  Level of Education by Age by Sex' respectively, of relevant persons.

	Then, we refer to the following paths:

		For ABS Census 'Highest Year of School Completed by Age by Sex' statistics:

			data/census/<ANY_STATE>/data/STE/<ANY_STATE>/<YEAR>Census_G16A_<ANY_STATE>_STE.csv
			data/census/<ANY_STATE>/data/STE/<ANY_STATE>/<YEAR>Census_G16B_<ANY_STATE>_STE.csv

		For ABS Census 'Non-School Qualification: Level of Education by Age by Sex' statistics:

			data/census/<ANY_STATE>/data/STE/<ANY_STATE>/<YEAR>Census_G46A_<ANY_STATE>_STE.csv
			data/census/<ANY_STATE>/data/STE/<ANY_STATE>/<YEAR>Census_G46B_<ANY_STATE>_STE.csv

	Separating these paths by state, we can determine 'Age-Gender-State-Education' correlation
'''

'''
	This function returns the total number of individuals surveyed by the ABS Census,
	by state, age range, gender, and education, in the language of the categories defined in the ASE.
'''
def interpret_abs_G16_G46():
	'''
		These fields are known from the 'Cell descriptors information', at 'Metadata_<YEAR>_GCP_DataPack.xlsx',
		and map to the ASE education types, as given below.
			
			PGrad_Deg -> Postgraduate degree level
			BachDeg -> Bachelor degree level
			GradDip_and_GradCert -> Year 12 or equivalent
			AdvDip_and_Dip -> Year 12 or equivalent
			Cert_III_IV -> Year 12 or equivalent
			Cert_I_II -> Less than year 12 or equivalent
			Cert_Levl_nfd -> Other
			Lev_Edu_IDes -> Other
			Lev_Edu_NS -> Prefer not to say

		Note:
		
			* Cert. 4 is equivalent to graduation of Year 12 : see https://www.tafensw.edu.au/study/pathways/tafe-to-university
			* "Courses at Diploma, Advanced Diploma and Associate degree level take between one and three years to complete, 
				and are generally considered to be equivalent to one to two years of study at degree level" : see
				https://en.wikipedia.org/wiki/Australian_Qualifications_Framework

		As we cannot adequately combine the fields for 'Other' and 'Prefer not to say' for tables 'G16' and 'G46', we do not include them
		in this data.

		The analysis proceeds that for table 'G46', all 'year 12 or equivalent' qualifications are discounted, as we will be retrieving
		this from table 'G16' instead.

		In saying this, we understand that members of table 'G16' that have completed 'year 12 or equivalent' education would naturally 
		also be quantified in post-schooling qualifications in table 'G46'. To remedy this overlap, we make the assumption that all members
		of post-schooling qualifications in table 'G46' would have had to complete 'year 12 or equivalent' education in order to obtain their
		qualifications.

		This means that we have to subtract the relevant values from table 'G46' from those of table 'G16' in order to distil the number
		of individuals who have completed 'year 12 or equivalent' education ONLY.
	'''
	# The table 'G46' only accounts for Bachelor and Postgraduate level degrees, in conjunction with the previously mentioned points
	ASE_BACKMAP_EDUCATION_G46 = {
		"PGrad_Deg" : "Postgraduate degree level",
		"BachDeg" : "Bachelor degree level",
		"AdvDip_and_Dip" : "Year 12 or equivalent",
		"Cert_III_IV" : "Year 12 or equivalent",
		"GradDip_and_GradCert" : "Year 12 or equivalent",
		"Cert_I_II" : "Less than year 12 or equivalent",
		"Cert_Levl_nfd" : "Undescribed",
		"Cert_Lev" : "Undescribed",
		"Lev_Edu_IDes" : "Undescribed",
		"Lev_Edu_NS" : "Undescribed",
		"Cumu" : "Cumulative"
	}
	ASE_BACKMAP_EDUCATION_G16 = {
		"Y12e" : "Year 12 or equivalent",
		"Y11e" : "Less than year 12 or equivalent",
		"Y10e" : "Less than year 12 or equivalent",
		"Y9e" : "Less than year 12 or equivalent",
		"Y8b" : "Less than year 12 or equivalent",
		"DNGTS" : "Less than year 12 or equivalent",
		"Hghst_yr_schl_ns" : "Undescribed",
		"Cumu" : "Cumulative"
	}

	'''
		Partial to this analysis, we require the ratios for which certain individuals of qualifications constitute the population.
		This is taken from the summary of results given for 'Level of highest educational attainment' at 
		https://quickstats.censusdata.abs.gov.au/census_services/getproduct/census/<YEAR>/quickstat/036
	'''
	CENSUS_HARDCODE_EDUCATION_ATTAINMENTS = {
		"Less than year 12 or equivalent" : 4687233,
		"Year 12 or equivalent" : 7675960,
		"Greater than Year 12 or equivalent" : 4181406,
		"Undescribed" : 1974794
	}

	CENSUS_HARDCODE_EDUCATION_ATTAINMENTS_BACKMAP = {
		"Year 12 or equivalent" : "Year 12 or equivalent",
		"Less than year 12 or equivalent" : "Less than year 12 or equivalent",
		"Bachelor degree level" : "Greater than Year 12 or equivalent",
		"Postgraduate degree level" : "Greater than Year 12 or equivalent",
		"Undescribed" : "Undescribed"
	}

	CENSUS_HARDCODE_EDUCATION_ATTAINMENTS_REMAPPER = {
		"Year 12 or equivalent": "y12_e",
		"Less than year 12 or equivalent": "lt_y12",
		"Postgraduate degree level": "gt_y12",
		"Bachelor degree level": "gt_y12"
	}

	####################

	ASE_BACKMAP_EDUCATION_G46 = {
		"PGrad_Deg" : "Tertiary",
		"BachDeg" : "Tertiary",
		"AdvDip_and_Dip" : "Secondary or less",
		"Cert_III_IV" : "Secondary or less",
		"GradDip_and_GradCert" : "Secondary or less",
		"Cert_I_II" : "Secondary or less",
		"Cert_Levl_nfd" : "Undescribed",
		"Cert_Lev" : "Undescribed",
		"Lev_Edu_IDes" : "Undescribed",
		"Lev_Edu_NS" : "Undescribed",
		"Cumu" : "Cumulative"
	}
	ASE_BACKMAP_EDUCATION_G16 = {
		"Y12e" : "Secondary or less",
		"Y11e" : "Secondary or less",
		"Y10e" : "Secondary or less",
		"Y9e" : "Secondary or less",
		"Y8b" : "Secondary or less",
		"DNGTS" : "Secondary or less",
		"Hghst_yr_schl_ns" : "Undescribed",
		"Cumu" : "Cumulative"
	}

	'''
		Partial to this analysis, we require the ratios for which certain individuals of qualifications constitute the population.
		This is taken from the summary of results given for 'Level of highest educational attainment' at 
		https://quickstats.censusdata.abs.gov.au/census_services/getproduct/census/<YEAR>/quickstat/036
	'''
	CENSUS_HARDCODE_EDUCATION_ATTAINMENTS = {
		"Less than year 12 or equivalent" : 4687233,
		"Year 12 or equivalent" : 7675960,
		"Greater than Year 12 or equivalent" : 4181406,
		"Undescribed" : 1974794
	}

	CENSUS_HARDCODE_EDUCATION_ATTAINMENTS_BACKMAP = {
		"Secondary or less" : "Year 12 or equivalent",
		"Secondary or less" : "Less than year 12 or equivalent",
		"Tertiary" : "Greater than Year 12 or equivalent",
		"Tertiary" : "Greater than Year 12 or equivalent",
		"Undescribed" : "Undescribed"
	}

	CENSUS_HARDCODE_EDUCATION_ATTAINMENTS_REMAPPER = {
		"Secondary or less": "y12_e",
		"Secondary or less": "lt_y12",
		"Tertiary" : "gt_y12"
	}


	state_age_gender_education_dict_G16 = abs_csv_to_dict(THIS_YEAR+"Census_G16", OMITTED_KEYS=["STE_CODE_"+THIS_YEAR])
	state_age_gender_education_dict_G46 = abs_csv_to_dict(THIS_YEAR+"Census_G49", OMITTED_KEYS=["STE_CODE_"+THIS_YEAR])
	'''
		We begin the analysis with the evaluation of individuals who either completed or did not complete year 12 education.
		This step requires that we first group the results by state, gender, and age.
	'''
	state_age_gender_y12e = dict()
	state_age_gender_y12e_c = dict()
	for this_state in state_age_gender_education_dict_G16.keys():
		for field, value in state_age_gender_education_dict_G16[this_state].items():
			# Set the adjusted field to reflect an adequate range for 85 and older.
			adjusted_field = field
			for abbrev in ["85ov", "85_ov", "85_ovr", "85_yrs_ovr", "85_85r"]:
				adjusted_field = adjusted_field.replace(abbrev, "85_85")
			# Set the cumulative field from the total
			adjusted_field = adjusted_field.replace("_yrs", "").replace("Tot_Tot", "Cumu")
			if ("Tot" not in adjusted_field):
				#print(adjusted_field)
				# Set the gender
				if (adjusted_field[0] in ASE_BACKMAP_GENDERS.keys()):
					this_gender = ASE_BACKMAP_GENDERS[adjusted_field[0]]
					#print(this_gender)
					#print(this_state)
					# Index the field for the qualification
					this_y12_qualification = None
					for code in ASE_BACKMAP_EDUCATION_G16.keys():
						if (code in adjusted_field):
							this_y12_qualification = ASE_BACKMAP_EDUCATION_G16[code]
							break
					# Set the age range
					this_age_range = "_".join(adjusted_field.split("_")[-2:])
					# If we are recording this field...
					if (this_y12_qualification is not None):
						if (this_y12_qualification != "Cumulative"):
								# ...Record the data
								if (this_state not in state_age_gender_y12e.keys()):
									state_age_gender_y12e[this_state] = dict()
								if (this_gender not in state_age_gender_y12e[this_state].keys()):
									state_age_gender_y12e[this_state][this_gender] = dict()
								if (this_y12_qualification not in state_age_gender_y12e[this_state][this_gender].keys()):
									state_age_gender_y12e[this_state][this_gender][this_y12_qualification] = dict()
								if (this_age_range not in state_age_gender_y12e[this_state][this_gender][this_y12_qualification].keys()):
									state_age_gender_y12e[this_state][this_gender][this_y12_qualification][this_age_range] = 0
								state_age_gender_y12e[this_state][this_gender][this_y12_qualification][this_age_range] += value
						else:
							# ...Record the associated cumulative data
							if (this_state not in state_age_gender_y12e_c.keys()):
								state_age_gender_y12e_c[this_state] = dict()
							if (this_gender not in state_age_gender_y12e_c[this_state].keys()):
								state_age_gender_y12e_c[this_state][this_gender] = dict()
							if (this_y12_qualification not in state_age_gender_y12e_c[this_state][this_gender].keys()):
								state_age_gender_y12e_c[this_state][this_gender] = 0
							state_age_gender_y12e_c[this_state][this_gender] += value
	# Print the data for 'stage-age-gender' correlations on Year 12 or lower educational qualifications
	#print(json.dumps(state_age_gender_y12e,indent=3))
	#print(json.dumps(state_age_gender_y12e_c,indent=3))
	state_age_gender_post12 = dict()
	state_age_gender_post12_c = dict()
	for this_state in state_age_gender_education_dict_G46.keys():
		for field, value in state_age_gender_education_dict_G46[this_state].items():
			# Set the adjusted field to reflect an adequate range for 85 and older.
			# There is an overlap with the matching for certificate level totals and male/female gender totals - we correct it here
			adjusted_field = field.replace("85ov", "85_85").replace("Lev_Tot_Total","Lev").replace("Tot_Total", "Cumu")
			if ("Tot" not in adjusted_field):
				# Set the gender
				if (adjusted_field[0] in ASE_BACKMAP_GENDERS.keys()):
					this_gender = ASE_BACKMAP_GENDERS[adjusted_field[0]]
					# Index the field for the qualification
					this_post12_qualification = None
					for code in (ASE_BACKMAP_EDUCATION_G46.keys()):
						if (code in adjusted_field):
							this_post12_qualification = ASE_BACKMAP_EDUCATION_G46[code]
					# ...Set the age range
					this_age_range = "_".join(adjusted_field.split("_")[-2:])
					if ("Cert_Lev" not in this_age_range):
						# If we are recording this field...
						if (this_post12_qualification is not None):
							# Totals in table G46 are recorded from the 'Total' keyword on each qualification
							if (this_post12_qualification != "Cumulative"):
								# ...Record the data
								if (this_state not in state_age_gender_post12.keys()):
									state_age_gender_post12[this_state] = dict()
								if (this_gender not in state_age_gender_post12[this_state].keys()):
									state_age_gender_post12[this_state][this_gender] = dict()
								if (this_post12_qualification not in state_age_gender_post12[this_state][this_gender].keys()):
									state_age_gender_post12[this_state][this_gender][this_post12_qualification] = dict()
								if (this_age_range not in state_age_gender_post12[this_state][this_gender][this_post12_qualification].keys()):
									state_age_gender_post12[this_state][this_gender][this_post12_qualification][this_age_range] = 0
								state_age_gender_post12[this_state][this_gender][this_post12_qualification][this_age_range] += value
							else:
								# ...Record the associated cumulative data
								if (this_state not in state_age_gender_post12_c.keys()):
									state_age_gender_post12_c[this_state] = dict()
								if (this_gender not in state_age_gender_post12_c[this_state].keys()):
									state_age_gender_post12_c[this_state][this_gender] = 0
								state_age_gender_post12_c[this_state][this_gender] += value
	# Print the data for 'stage-age-gender' correlations on post-Year 12 qualifications
	#print(json.dumps(state_age_gender_post12,indent=3))
	#print(json.dumps(state_age_gender_post12_c,indent=3))
	'''
		In this step, we take each dataset, relative to 'state', 'gender', and 'age', and we map the distribution of 
		values collected for the education levels recorded in the ABS Census, to the age ranges of ASE. This is achieved by finding the 
		interpolation of Census values at the midpoint of each ASE education level range, as is calculated below:
	'''
	for dataset in [state_age_gender_y12e, state_age_gender_post12]:
		for this_state in dataset.keys():
			for this_gender in dataset[this_state].keys():
				for this_qualification in dataset[this_state][this_gender].keys():
					# Get the dataset, and create a copy
					census_education_midpoint_values_dataset = dataset[this_state][this_gender][this_qualification].copy()
					# Formalise the age midpoints of the census...
					age_midpoints = [midpoint(int(x.split("_")[0]), int(x.split("_")[1])) for x in census_education_midpoint_values_dataset.keys()]
					# ...and formalise the associated values as well...
					age_values = [x for x in census_education_midpoint_values_dataset.values()]
					# ...and also formalise the ASE age midpoints
					ase_age_midpoints = [midpoint(x[0], x[-1]) for x in ASE_BACKMAP_AGE.values()]
					'''
						We deliberately set the last midpoint of the ASE age midpoints to only the first index of its range, to avoid any interpolation 
						cutoff issues
					'''
					ase_age_midpoints[-1] = list(ASE_BACKMAP_AGE.values())[-1][0]
					# Then we calculate the values, as they would've been for the ASE
					new_values = spline_projection_from_frequencies(age_values, age_midpoints, ase_age_midpoints)
					# Finally, we reinsert the values into the original dictionary
					dataset[this_state][this_gender][this_qualification] = { 
														list(ASE_BACKMAP_AGE.keys())[i] : new_values[i] for i in range(len(ASE_BACKMAP_AGE)) }
	'''
		In this step, we combine the datasets and normalise them to the amounts declared in the CENSUS_HARDCODE_EDUCATION_ATTAINMENTS
		dictionary; this is done by firstly regrouping the data to feature the qualification as the key of the shallowest depth. Both dataset
		for 'pre - Year 12 or equivalent' and 'post-Year 12' are combined, and we keep records of the totaled amounts for each qualification across
		age, gender, and state.
	'''
	state_age_gender_education_dict = dict()
	state_age_gender_education_dict_totals = dict()
	for dataset in [state_age_gender_y12e, state_age_gender_post12]:
		for this_state in dataset.keys():
			for this_gender in dataset[this_state].keys():
				for this_qualification in dataset[this_state][this_gender].keys():
					for this_age_range, value in dataset[this_state][this_gender][this_qualification].items():
						if (this_qualification not in state_age_gender_education_dict.keys()):
							state_age_gender_education_dict_totals[this_qualification] = 0
							state_age_gender_education_dict[this_qualification] = dict()
						if (this_state not in state_age_gender_education_dict[this_qualification].keys()):
							state_age_gender_education_dict[this_qualification][this_state] = dict()
						if (this_gender not in state_age_gender_education_dict[this_qualification][this_state].keys()):
							state_age_gender_education_dict[this_qualification][this_state][this_gender] = dict()
						if (this_age_range not in state_age_gender_education_dict[this_qualification][this_state][this_gender].keys()):
							state_age_gender_education_dict[this_qualification][this_state][this_gender][this_age_range] = 0
						state_age_gender_education_dict[this_qualification][this_state][this_gender][this_age_range] += value
						# Apply the amount to the total
						state_age_gender_education_dict_totals[this_qualification] += value
	'''
	# Print the combined 'stage-age-gender' correlations for all education qualifications, and associated totals
	print(json.dumps(state_age_gender_education_dict,indent=3))
	print(json.dumps(state_age_gender_education_dict_totals,indent=3))
	'''
	'''
		Normalise the values to reflect the CENSUS_HARDCODE_EDUCATION_ATTAINMENTS dictionary
	'''
	for this_qualification in state_age_gender_education_dict.keys():
		for this_state in state_age_gender_education_dict[this_qualification].keys():
			for this_gender in state_age_gender_education_dict[this_qualification][this_state].keys():
				for this_age_range in state_age_gender_education_dict[this_qualification][this_state][this_gender].keys():
					v = state_age_gender_education_dict[this_qualification][this_state][this_gender][this_age_range]
					v /= state_age_gender_education_dict_totals[this_qualification]
					v *= CENSUS_HARDCODE_EDUCATION_ATTAINMENTS[CENSUS_HARDCODE_EDUCATION_ATTAINMENTS_BACKMAP[this_qualification]]
					state_age_gender_education_dict[this_qualification][this_state][this_gender][this_age_range] = v
	'''
	# Print the distribution of individuals surveyed by the census, by state, age, gender, and education
	print(json.dumps(state_age_gender_education_dict,indent=3))
	'''
	return state_age_gender_education_dict

'''
	State-Party Preference Correlation

	* We refer to the latest Electoral Comission results for 'First preferences by state by party':
		2016 : https://results.aec.gov.au/20499/Website/HouseDownloadsMenu-20499-Csv.htm
		2022 : https://results.aec.gov.au/27966/Website/HouseDownloadsMenu-27966-Csv.htm
'''
def interpret_ecomm():
	lines = open(os.path.join(os.getcwd(), "data", "electoral_commission", "HouseFirstPrefsByStateByPartyDownload-27966.csv")).read().split("\n")
	state_party = dict()
	for i in range(len(lines)):
		# First 2 lines are headings - last line is padding
		if (i >= 2) and (i < len(lines)-1):
			vals = lines[i].split(",")
			party = vals[1]
			# State
			if (vals[0] not in state_party.keys()):
				state_party[vals[0]] = dict()
			# State -> Party
			if (party not in ELECTORAL_COMMISSION_CODES.keys()):
				party = "Undescribed"
			if (party not in state_party[vals[0]].keys()):
				state_party[vals[0]][party] = 0
			state_party[vals[0]][party] += int(vals[7])

	parties = list()
	[parties.extend(list(state_party[x].keys())) for x in state_party.keys()]
	parties = set(parties)

	# Combine the LNP and LP, and pad all states with representation of all parties
	for state in state_party.keys():
		# Declare the LNP if it doesn't exist, and then add the LP to it
		if ("LNP" not in state_party[state]):
			state_party[state]["LNP"] = 0
		if ("LP" in state_party[state]):
			state_party[state]["LNP"] += state_party[state]["LP"]
			del state_party[state]["LP"]
		for party in ELECTORAL_COMMISSION_CODES:
			if (party not in state_party[state].keys() and (party != "LP")):
				state_party[state][party] = 0
	#print(json.dumps(state_party,indent=3))
	#sys.exit()
	return state_party


'''
master_dict = interpret_ecomm()
print(json.dumps(master_dict,indent=3))
#print(json.dumps(,indent=3))
with open("Joint_Distribution_Australian_Population_State_Party_Preference.csv", "w") as f:
	f.write(pd.DataFrame({(i): master_dict[i]
										for i in master_dict.keys()}).to_csv())
	f.close()
sys.exit()
'''
'''
	
	start with the state-age-gender-employment correlation

			use the education to employment ratios to 

'''
def construct_category_correlations():
	total_of_abs_participants = 0
	# Start with the state-age-gender-employment-correlations
	correlation_dict = interpret_abs_G43_or_G46()
	# Apply education
	correlation_dict_education = interpret_abs_G16_G46()
	for this_state in correlation_dict.keys():
		for this_gender in correlation_dict[this_state].keys():
			for this_employment in correlation_dict[this_state][this_gender].keys():
				for this_age_range in correlation_dict[this_state][this_gender][this_employment].keys():
					total_members_in_cohort = correlation_dict[this_state][this_gender][this_employment][this_age_range]
					'''
						While we can apply a 'education-employment' correlation here to determine how to distribute values among both
						categories, it will not be necessary as the employment has already distributed its values.

						Instead, when determining the values for 'education-employment' correlation, we retrieve the education distribution under
						general terms for state, gender, and age range, and normalise these values to the amount provided by the employment value.
					'''
					isolated_education_dict = {k:v[this_state][this_gender][this_age_range] for k,v in correlation_dict_education.items()}
					total_isolated_education_dict = sum(isolated_education_dict.values())
					correlation_dict[this_state][this_gender][this_employment][this_age_range] = {k:v/total_isolated_education_dict*total_members_in_cohort for k,v in isolated_education_dict.items()}
					total_of_abs_participants += sum(correlation_dict[this_state][this_gender][this_employment][this_age_range].values())
	# Apply income
	correlation_dict_income = interpret_abs_G17()
	for this_state in correlation_dict.keys():
		for this_gender in correlation_dict[this_state].keys():
			for this_employment in correlation_dict[this_state][this_gender].keys():
				for this_age_range in correlation_dict[this_state][this_gender][this_employment].keys():
					for this_qualification in correlation_dict[this_state][this_gender][this_employment][this_age_range].keys():
						total_members_in_cohort = correlation_dict[this_state][this_gender][this_employment][this_age_range][this_qualification]
						isolated_income_dict = correlation_dict_income[this_state][this_gender][this_age_range]
						total_isolated_income_dict = sum(isolated_income_dict.values())
						correlation_dict[this_state][this_gender][this_employment][this_age_range][this_qualification] = {k:v/total_isolated_income_dict*total_members_in_cohort for k,v in isolated_income_dict.items()}
	
	correlation_dict_party = interpret_ecomm()

	'''
		Apply party preference and normalise
	'''
	preadjusted_total = 0
	for this_state in correlation_dict.keys():
		for this_gender in correlation_dict[this_state].keys():
			for this_employment in correlation_dict[this_state][this_gender].keys():
				for this_age_range in correlation_dict[this_state][this_gender][this_employment].keys():
					for this_qualification in correlation_dict[this_state][this_gender][this_employment][this_age_range].keys():
						for this_income in correlation_dict[this_state][this_gender][this_employment][this_age_range][this_qualification].keys():
							total_members_in_cohort = correlation_dict[this_state][this_gender][this_employment][this_age_range][this_qualification][this_income]
							isolated_party_dict = correlation_dict_party[this_state]
							total_isolated_party_dict = sum(isolated_party_dict.values())

							new_dict = dict()
							for k,v in isolated_party_dict.items():
								if (ELECTORAL_COMMISSION_CODES[k] not in new_dict.keys()):
									new_dict[ELECTORAL_COMMISSION_CODES[k]] = 0
								new_dict[ELECTORAL_COMMISSION_CODES[k]] += (v/total_isolated_party_dict*total_members_in_cohort/total_of_abs_participants)

							correlation_dict[this_state][this_gender][this_employment][this_age_range][this_qualification][this_income] = new_dict
							preadjusted_total += sum(correlation_dict[this_state][this_gender][this_employment][this_age_range][this_qualification][this_income].values())
	
	for this_state in correlation_dict.keys():
		for this_gender in correlation_dict[this_state].keys():
			for this_employment in correlation_dict[this_state][this_gender].keys():
				for this_age_range in correlation_dict[this_state][this_gender][this_employment].keys():
					for this_qualification in correlation_dict[this_state][this_gender][this_employment][this_age_range].keys():
						for this_income in correlation_dict[this_state][this_gender][this_employment][this_age_range][this_qualification].keys():
							correlation_dict[this_state][this_gender][this_employment][this_age_range][this_qualification][this_income] = \
								{k:v/preadjusted_total for k,v in correlation_dict[this_state][this_gender][this_employment][this_age_range][this_qualification][this_income].items()}
	return correlation_dict

if (__name__ == "__main__"):
	master_dict = construct_category_correlations()

	with open(f"census{THIS_YEAR}_australian_population_representation.json","w") as f:
		f.write(json.dumps(master_dict,indent=3))
		f.close()

	with open(f"Joint_Distribution_Australian_Population_Age_Gender_State_Education_Income_Party_Preference_{THIS_YEAR}.csv", "w") as f:
		f.write(pd.DataFrame({(i,j,k,l,m,n): master_dict[i][j][k][l][m][n]
											for i in master_dict.keys() 
											for j in master_dict[i].keys()
											for k in master_dict[i][j].keys()
											for l in master_dict[i][j][k].keys()
											for m in master_dict[i][j][k][l].keys()
											for n in master_dict[i][j][k][l][m].keys()}).to_csv())
		f.close()

master_dict = json.loads(open(f"census{THIS_YEAR}_australian_population_representation.json").read())

def distribution_state_of_residence(master_dict):
	distributions = {k:dict() for k in [
		"STATE_OF_RESIDENCE", "GENDER", "EMPLOYMENT_STATUS", "AGE", 
		"EDUCATION_LEVEL", "INCOME_BRACKET", "POLITICAL_PARTY_PREFERENCE"]}
	#
	for state_of_residence in master_dict:
		for gender in master_dict[state_of_residence]:
			for employment_status in master_dict[state_of_residence][gender]:
				for age in master_dict[state_of_residence][gender][employment_status]:
					for education_level in master_dict[state_of_residence][gender][employment_status][age]:
						for income_bracket in master_dict[state_of_residence][gender][employment_status][age][education_level]:
							for political_party_preference in master_dict[state_of_residence][gender][employment_status][age][education_level][income_bracket]:
								v = master_dict[state_of_residence][gender][employment_status][age][education_level][income_bracket][political_party_preference]
								# Compile data
								dc_to_v = {
										"STATE_OF_RESIDENCE" : state_of_residence,
										"GENDER" : gender,
										"EMPLOYMENT_STATUS" : employment_status,
										"AGE" : age,
										"EDUCATION_LEVEL" : education_level,
										"INCOME_BRACKET" : income_bracket,
										"POLITICAL_PARTY_PREFERENCE" : political_party_preference
									}
								for dc in dc_to_v:
									if (not dc_to_v[dc] in distributions[dc]):
										distributions[dc][dc_to_v[dc]] = float()
									distributions[dc][dc_to_v[dc]] += v
	return distributions
'''
	with open("abs_distributions.json", "w") as f:
		f.write(json.dumps(distribution_state_of_residence(master_dict),indent=3))
		f.close()
'''
distributions_titles_mapped = {
		"STATE_OF_RESIDENCE" : "State Of Residence",
		"GENDER" : "Gender",
		"EMPLOYMENT_STATUS" : "Employment Status",
		"AGE" : "Age (excluding individuals under 18)",
		"EDUCATION_LEVEL" : "Education Level",
		"INCOME_BRACKET" : "Income Bracket",
		"POLITICAL_PARTY_PREFERENCE" : "Political Party Preference"
	}


distributions_responses_mapped = {
		"STATE_OF_RESIDENCE" : ["QLD", "NSW", "VIC", "ACT", "NT", "WA", "TAS", "SA"],
		"GENDER" : ["Male", "Female"],
		"EMPLOYMENT_STATUS" : ["Employed", "Retired", "Unemployed", "Undescribed"],
		"AGE" : ["18 - 34", "35 - 64", "65 and over"],
		"EDUCATION_LEVEL" : ["Secondary or less", "Tertiary", "Undescribed"],
		"INCOME_BRACKET" : ["$1 - $51,999", "$52,000 - $103,999", "$104,000 - $155,999", "Undescribed"],
		"POLITICAL_PARTY_PREFERENCE" : ["Australian Labor Party", "Liberal National Party", "Australian Greens", "Undescribed"]
	}

def visualise_distribution_for_demographic_characteristic(dc, distributions):
	fig, ax = plt.subplots(1)
	fig.set_size_inches(8,3.5)
	ax.bar(distributions_responses_mapped[dc], height=[distributions[dc][x] for x in distributions_responses_mapped[dc]], width=0.7, color="#557a9e")
	ax.set_ylabel("Percentage (%)")
	ax.set_yticks([x/100 for x in range(0,round(max(distributions[dc].values())*120), 
											round(max(distributions[dc].values())*20))])
	ax.set_yticklabels([f"{x}%" for x in range(0,round(max(distributions[dc].values())*120), 
													round(max(distributions[dc].values())*20))])
	ax.set_xticklabels(distributions_responses_mapped[dc],rotation=45)
	ax.grid()
	ax.set_axisbelow(True)
	ax.set_xlabel("Response")
	ax.set_title(f"Percentage Of Australians By {distributions_titles_mapped[dc]}")


