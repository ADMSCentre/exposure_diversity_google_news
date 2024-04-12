import json
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = "Times New Roman"
'''
	The purpose of this code is to map various incoming user data types (from the ABS Census, 
	Australian Search Experience, Australian Ad Observatory), relating demographic information to a single
	unified schema.
'''

# Load in the demographic mappings dictionary
demographic_mappings = json.loads(open("demographic_mappings.json").read())

'''
	This function interprets Austrlian states & territories from postcodes
'''
def postcode_to_state(val):
	if (val is None):
		return None
	val_i = int(val)
	for x in [[1000,1999], [2000,2599], [2619,2899], [2921,2999]]:
		if (val_i >= x[0]) and (val_i <= x[1]):
			return "NSW"
	for x in [[200,299], [2600,2618], [2900,2920]]:
		if (val_i >= x[0]) and (val_i <= x[1]):
			return "ACT"
	for x in [[3000,3999], [8000,8999]]:
		if (val_i >= x[0]) and (val_i <= x[1]):
			return "VIC"
	for x in [[4000,4999], [9000,9999]]:
		if (val_i >= x[0]) and (val_i <= x[1]):
			return "QLD"
	for x in [[5000,5799], [5800,5999]]:
		if (val_i >= x[0]) and (val_i <= x[1]):
			return "SA"
	for x in [[6000,6797], [6800,6999]]:
		if (val_i >= x[0]) and (val_i <= x[1]):
			return "WA"
	for x in [[7000,7799], [7800,7999]]:
		if (val_i >= x[0]) and (val_i <= x[1]):
			return "TAS"
	for x in [[800,899], [900,999]]:
		if (val_i >= x[0]) and (val_i <= x[1]):
			return "NT"
	return None

'''
	Map a user's value for a given category to the accepted values of the category within the
	demographic mappings dictionary
'''
def map_demographic_category(category, user, demographic_mappings=demographic_mappings):
	# Retrieve the mappings for the category
	mappings = demographic_mappings[category]
	# Preserve the field
	if ("preserve" in mappings.keys()):
		if (user[category] in mappings["preserve"]):
			return user[category]
	# Map the field
	if ("map" in mappings.keys()):
		if (user[category] in mappings["map"].keys()):
			return mappings["map"][user[category]]
	# If all else fails, return the default value
	return demographic_mappings["default"]


'''
	Load in the 'mappable' details of all users from a static JSON-delimited file
'''
def load_participating_cohort_of_australian_search_experience():
	output_user_dict = dict()
	# Load in the user details
	# For every user...

	contributing_activation_codes = json.loads(open("contributing_activation_codes.json").read())
	for user in [json.loads(x) for x in open("australian_search_experience_participants.json").read().split("\n") if (len(x) > 0)]:
		if user["activation_code"] in contributing_activation_codes:
			output_user_dict_user = dict()
			# Map the details
			user["employment"] = user["employment_status"]
			user["education"] = user["level_education"]
			user["income"] = user["income_bracket"]
			user["party"] = user["party_preference"]
			user["state"] = postcode_to_state(user["postcode"])
			output_user_dict_user["state"] = map_demographic_category("state", user)
			output_user_dict_user["gender"] = map_demographic_category("gender", user)
			output_user_dict_user["employment"] = map_demographic_category("employment", user)
			output_user_dict_user["age"] = map_demographic_category("age", user)
			output_user_dict_user["education"] = map_demographic_category("education", user)
			output_user_dict_user["income"] = map_demographic_category("income", user)
			output_user_dict_user["party"] = map_demographic_category("party", user)
			output_user_dict[user["id"]] = output_user_dict_user
	return output_user_dict

def list_of_json_to_weighted_dict():
	weighted_dict = dict()
	participant_cohort = load_participating_cohort_of_australian_search_experience()
	for user_activation_code in participant_cohort:
		for dc in participant_cohort[user_activation_code]:
			v = participant_cohort[user_activation_code][dc]
			if (not dc in weighted_dict):
				weighted_dict[dc] = dict()
			if (not v in weighted_dict[dc]):
				weighted_dict[dc][v] = int()
			weighted_dict[dc][v] += 1
	for dc in weighted_dict:
		this_dc_sum = sum(weighted_dict[dc].values())
		for v in weighted_dict[dc]:
			weighted_dict[dc][v] /= this_dc_sum
	return weighted_dict

this_demographic_characteristic_secondary_mappings = {
		"gender" : "GENDER",
		"state" : "STATE_OF_RESIDENCE",
		"employment" : "EMPLOYMENT_STATUS",
		"age" : "AGE",
		"education" : "EDUCATION_LEVEL",
		"income" : "INCOME_BRACKET",
		"party" : "POLITICAL_PARTY_PREFERENCE"
	}


'''
	This function plots the comparison of the data obtained through the Australian Search Experience, against the Australian Bureau of Statistics data
'''
def comparison_of_representation(abs_distributions, ase_distributions, distributions_responses_mapped, distributions_titles_mapped, this_demographic_characteristic):
	this_demographic_characteristic_secondary = this_demographic_characteristic_secondary_mappings[this_demographic_characteristic]
	ordered_responses = distributions_responses_mapped[this_demographic_characteristic_secondary]
	ordered_responses_values = [ase_distributions[this_demographic_characteristic][x] 
								for x in distributions_responses_mapped[this_demographic_characteristic_secondary]]
	ordered_responses_values_b = [abs_distributions[this_demographic_characteristic_secondary][x] 
								  for x in distributions_responses_mapped[this_demographic_characteristic_secondary]]
	ind = np.arange(len(ordered_responses))
	fig, ax = plt.subplots(1)
	fig.set_size_inches(10,3.5) 
	width = 0.35	
	ax.bar(ind, ordered_responses_values , width, label='Australian Search Experience', color="#557a9e")
	ax.bar(ind + width, ordered_responses_values_b, width, label='Australian Bureau Of Statistics', color="#ad4953")
	max_v = max(ordered_responses_values)
	max_v = max([max_v] + ordered_responses_values_b)
	ax.set_yticks([x/100 for x in range(0,round(max_v*120), round(max_v*20))])
	ax.set_yticklabels([f"{x}%" for x in range(0,round(max_v*120), round(max_v*20))])
	ax.set_xlabel('Response')
	ax.set_ylabel('Percentage (%)')
	ax.set_title("Comparison Of Representation - " + distributions_titles_mapped[this_demographic_characteristic_secondary])
	ax.set_xticks(ind + width / 2, ordered_responses)
	ax.set_xticklabels(ordered_responses,rotation=90)
	ax.grid()
	ax.set_axisbelow(True)
	ax.legend(loc='best')
	plt.show()

'''
	This function plots the distribution of the spoken languages
'''
def distribution_of_languages():
	contributing_activation_codes = json.loads(open("contributing_activation_codes.json").read())
	reported_languages = ["Other" if (x["language"] is None) else (x["language"] 
				if (x["language"] != "Other") else x["language_specify"])  
		 for x in [json.loads(y) for y in 
				open("australian_search_experience_participants.json").read().split("\n") if (len(y) > 0)] 
						  if (x["activation_code"] in contributing_activation_codes)]
	weighted_dict = dict()
	for x in reported_languages:
		if (not x in weighted_dict):
			weighted_dict[x] = int()
		weighted_dict[x] += 1

	fig, ax = plt.subplots(1)
	fig.set_size_inches(10,3.5) 
	ax.bar(weighted_dict.keys(), height=weighted_dict.values(), color="#557a9e")
	ax.set_xticklabels(weighted_dict.keys(),rotation=90)
	ax.grid()
	ax.set_axisbelow(True)
	ax.set_xlabel('Language')
	ax.set_ylabel('No. Of Participants')
	ax.set_title("Distribution Of Languages Spoken Among Participants")
	plt.show()

'''
	This function loads the participants of the Australian Search Experience, prior to undertaking any mapping
'''
def load_participating_cohort_of_australian_search_experience_unmapped():
	contributing_activation_codes = json.loads(open("contributing_activation_codes.json").read())
	participants = [x for x in [json.loads(y) for y in open("australian_search_experience_participants.json").read().split("\n") if (len(y) > 0)] if (x["activation_code"] in contributing_activation_codes)]
	weighted_dict = dict()
	for participant in participants:
		for dc in participant:
			if (dc in ["age", "employment_status", "gender", "party_preference", "income_bracket", "postcode", "level_education"]):
				applied_dc = dc
				if (dc == "postcode"):
					applied_dc = "state"
				if (not applied_dc in weighted_dict):
					weighted_dict[applied_dc] = dict()
				applied_v = participant[dc]
				if (applied_dc == "state"):
					applied_v = postcode_to_state(applied_v)
				if (applied_v is None):
					applied_v = "Prefer not to say"
				if (not applied_v in weighted_dict[applied_dc]):
					weighted_dict[applied_dc][applied_v] = int()
				weighted_dict[applied_dc][applied_v] += 1
	return weighted_dict

def distribution_of_demographic_characteristic_unmapped(participating_cohort_of_australian_search_experience_unmapped, dc):
	values_mapped = {
			"age" : ['18 - 24', '25 - 34', '35 - 44', '45 - 54', '55 - 64', '65 - 74', '75 and over', 'Prefer not to say'],
			"gender" : ['Male', 'Female', 'Other', 'Prefer not to say'],
			"employment_status" : ['Employed full-time', 'Employed part-time', 'Unemployed and looking for work', 'Unemployed and not looking for work', 'Retired', 'Prefer not to say'],
			"state" : ['VIC', 'NSW', 'TAS', 'QLD', 'SA', 'ACT', 'WA', 'NT'],
			"income_bracket" : ['$1 - $15,599', '$15,600 - $20,799', '$20,800 - $25,999', '$26,000 - $33,799', '$33,800 - $41,599', '$41,600 - $51,999', '$52,000 - $64,999', '$65,000 - $77,999', '$78,000 - $90,999', '$91,000 - $103,999', '$104,000 - $155,999','$156,000 or more', 'Prefer not to say'],
			"party_preference" : ['Labor', 'Greens', 'Liberal', 'National', 'One Nation', 'Other', 'None', 'Prefer not to say'],
			"level_education" : ['Less than year 12 or equivalent', 'Year 12 or equivalent', 'Bachelor degree level', 'Postgraduate degree level', 'Prefer not to say']
			}

	values_mapped_names = {
			"age" : {
					'18 - 24' : '18 - 24',
					'25 - 34' : '25 - 34',
					'35 - 44' : '35 - 44',
					'45 - 54' : '45 - 54',
					'55 - 64' : '55 - 64',
					'65 - 74' : '65 - 74',
					'75 and over' : '75+',
					'Prefer not to say' : 'Undescribed'
				},
			"gender" : {
					'Male' : 'Male',
					'Female' : 'Female',
					'Other' : 'Other',
					'Prefer not to say' : 'Undescribed'
				},
			"employment_status" : {
					'Employed full-time' : 'Full-Time',
					'Employed part-time' : 'Part-Time',
					'Unemployed and looking for work' : 'Looking',
					'Unemployed and not looking for work' : 'Not Looking',
					'Retired' : 'Retired',
					'Prefer not to say' : 'Undescribed'
				},
			"state" : {
					'VIC' : 'VIC',
					'NSW' : 'NSW',
					'TAS' : 'TAS',
					'QLD' : 'QLD',
					'SA' : 'SA',
					'ACT' : 'ACT',
					'WA' : 'WA',
					'NT' : 'NT'
				},
			"income_bracket" : {
					'$1 - $15,599' : '$1 - $15,599',
					'$15,600 - $20,799' : '$15,600 - $20,799',
					'$20,800 - $25,999' : '$20,800 - $25,999',
					'$26,000 - $33,799' : '$26,000 - $33,799',
					'$33,800 - $41,599' : '$33,800 - $41,599',
					'$41,600 - $51,999' : '$41,600 - $51,999',
					'$52,000 - $64,999' : '$52,000 - $64,999',
					'$65,000 - $77,999' : '$65,000 - $77,999',
					'$78,000 - $90,999' : '$78,000 - $90,999',
					'$91,000 - $103,999' : '$91,000 - $103,999',
					'$104,000 - $155,999' : '$104,000 - $155,999',
					'$156,000 or more' : '$156,000+',
					'Prefer not to say' : 'Undescribed'
				},
			"party_preference" : {
					'Labor' : 'Labor',
					'Greens' : 'Greens',
					'Liberal' : 'Liberal',
					'National' : 'National',
					'One Nation' : 'One Nation',
					'Other' : 'Other',
					'None' : 'None',
					'Prefer not to say' : 'Undescribed'
				},
			"level_education" : {
					'Less than year 12 or equivalent' : 'Less than Yr. 12',
					'Year 12 or equivalent' : 'Yr. 12',
					'Bachelor degree level' : 'Bachelor Deg.',
					'Postgraduate degree level' : 'Postgraduate Deg.',
					'Prefer not to say' : 'Undescribed',
				}
			}
	titles_mapped = {
			"age" : "Age",
			"gender" : "Gender",
			"employment_status" : "Employment Status",
			"level_education" : "Level Of Education",
			"state" : "State Of Residence",
			"income_bracket" : "Income Bracket",
			"party_preference" : "Political Party Preference"
		}
	fig, ax = plt.subplots(1)
	fig.set_size_inches(5,3.5) 
	ax.bar(values_mapped[dc], [participating_cohort_of_australian_search_experience_unmapped[dc][x] for x in values_mapped[dc]], color="#557a9e")
	if (dc in ["income_bracket", "party_preference", "level_education"]):
		ax.set_xticklabels([values_mapped_names[dc][x] for x in values_mapped[dc]], rotation=90)
	else:
		ax.set_xticklabels([values_mapped_names[dc][x] for x in values_mapped[dc]])
	ax.grid()
	ax.set_axisbelow(True)
	ax.set_xlabel(titles_mapped[dc])
	ax.set_ylabel('No. Of Participants')
	ax.set_title(f"Distribution Of {titles_mapped[dc]} Among Participants")
	plt.gcf().subplots_adjust(bottom=0.40)
	plt.subplots_adjust(bottom=0.40)
	plt.tight_layout()
	plt.savefig(f"Distribution Of {titles_mapped[dc]} Among Participants.png", dpi=200)
	plt.show()



