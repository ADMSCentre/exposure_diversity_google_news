'''
	This script is responsible for the implementation of the demographic analysis.

	For the execution of the demographic analysis, the following files are directly obtained from the 'Publisher Analysis':
		* output.jsond.temporal.k1.p4
		* output.jsond.temporal.k2.p4
		* entries_by_domain
		* entries_by_domain_summarised
'''
import os
import json
import ipdb
import math
import traceback
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
import warnings

GLOBAL_SHOW = True

warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = "Times New Roman"

# Participating cohort
participants = {x["activation_code"]:x for x in json.loads(open("participating_cohort.json").read())}
# Load in the cohort statistics for the participating cohort
participating_cohort_statistics = json.loads(open("participating_cohort_statistics.json").read())
# Load in the compressed dataset of Google News data donations for the designated time window
keyword_types = ["k1", "k2"]


# Load in the entries_by_domain file
entries_by_domain = json.loads(open(f"entries_by_domain.json").read())

entries_by_domain_summarised = json.loads(open(f"entries_by_domain_summarised.json").read())

# Specify the various demographic types
demographic_types = ["gender", "age", "postcode", "level_education", "employment_status", 
											"party_preference", "income_bracket",  "language"]
# Specify the postcode mappings
postcode_mappings = json.loads(open("postcode_mappings.json").read())
# The demographic details of the participating cohort of individuals for the designated time window.
participating_cohort = json.loads(open("participating_cohort.json").read())
# The list of activation codes that pertain to the individuals who contributed data donations for the relevant time window.
contributing_activation_codes = json.loads(open("contributing_activation_codes.json").read())


# Load up the contextualised set, for the first set of keywords that were evaluated
contextualised_set_k1 = json.loads(open("contextualised_set.k1.json").read())

# Load up the contextualised set, for the second set of keywords that were evaluated
contextualised_set_k2 = json.loads(open("contextualised_set.k2.json").read())

contextualised_set_summarised = json.loads(open("contextualised_set_summarised.json").read())

contextualised_set = {
		"k1" : contextualised_set_k1,
		"k2" : contextualised_set_k2
	}

# Load up the uncut nondistinct contextualised set, for the both sets of keywords that were evaluated
#contextualised_set_uncut_nondistinct = {k:json.loads(open(f"keyword_domain_demographic_breakdowns.uncut.nondistinct.{k}.json").read()) for k in ["k1", "k2"]}

	
'''
	This function determines the state of an individual, based on a provided postcode
'''
def determine_state(postcode, postcode_mappings=postcode_mappings):
	for state in postcode_mappings:
		found = False
		for this_range in postcode_mappings[state]:
			if (int(postcode) in list(range(this_range[0], this_range[1]))):
				found = True
				break
		if (found):
			return state
	return "Unknown"

'''
	Given a set of activation codes, this function determines the breakdown of demographic statistics
	for said set.
'''
def participant_breakdown(participating_cohort, demographic_types=demographic_types):
	statistics = dict()
	# For each demographic type
	for d in demographic_types:
		# For each member of the provided participating cohort
		for u in participating_cohort:
			v_name =  u[d]
			if (d == "postcode"):
				v_name = determine_state(v_name)
			if (d not in statistics):
				statistics[d] = dict()
			if (v_name not in statistics[d]):
				statistics[d][v_name] = int()
			statistics[d][v_name] += 1
	return statistics

'''
	This function filters the participants by their relevant activation codes.
'''
def participant_breakdown_filtered(contributing_activation_codes, participating_cohort=participating_cohort):
	return [x for x in participating_cohort if (x["activation_code"] in contributing_activation_codes)]

'''
	This function compiles the 'participating_cohort_statistics.json' file, which summarises the frequencies
	of the various individuals who submitted data donations across the investigated window of time, segmented by relevant
	demographic characteristics.
'''
def participant_participating_cohort_statistics():
	with open(f"participating_cohort_statistics.json", "w") as f:
		f.write(json.dumps(participant_breakdown(
			participant_breakdown_filtered(contributing_activation_codes)), indent=3))
		f.close()

'''
	This function produces a dictionary that can be used to map Australian postcodes 
	to their respective states/territories
'''
def postcodes_to_demographic_characteristic_states():
	# Load in the 'state <-> postcode' mappings dictionary
	postcode_state_mappings = json.loads(open("postcode_mappings.json").read())
	postcode_state_mappings_rendered = dict()
	for k,v in postcode_state_mappings.items():
		# For each range
		for this_range in v:
			# Apply it to the relative key (being the necessary state/territory)
			if (k not in postcode_state_mappings_rendered.keys()):
				postcode_state_mappings_rendered[k] = list()
			postcode_state_mappings_rendered[k].extend(list(range(this_range[0], this_range[1]+1)))
	return postcode_state_mappings_rendered

postcode_state_mappings_rendered = postcodes_to_demographic_characteristic_states()

def postcode_to_state(postcode, postcode_state_mappings=postcode_state_mappings_rendered):
	# Run postcode -> state conversion
	for this_state, this_postcode_range in postcode_state_mappings.items():
		if (int(postcode) in this_postcode_range):
			return this_state
	return None

'''
	This function generates the contributing activation codes for the designated time window.
'''
def generate_contributing_activation_codes():
	# Determine contributors overall
	contributing_activation_codes = list()
	# Gather...
	distinct_type = "distinct"
	for keyword_category_set in entries_by_domain[distinct_type]:
		for keyword in entries_by_domain[distinct_type][keyword_category_set]:
			for domain in entries_by_domain[distinct_type][keyword_category_set][keyword]:
				for entry in entries_by_domain[distinct_type][keyword_category_set][keyword][domain]:
					contributing_activation_codes.append(entry["activation_code"])
	contributing_activation_codes = list(set(contributing_activation_codes))
	with open(f"contributing_activation_codes.json", "w") as f:
		f.write(json.dumps(contributing_activation_codes, indent=3))
		f.close()

'''
	This function generates the breakdown of the various demographics, segmented by domains and keywords
'''
def generate_keyword_domain_demographic_breakdowns(cut=False):
	distinct_type = "nondistinct"
	# Contributors per domain
	keyword_domain_demographic_breakdowns = dict()
	for keyword_category_set in entries_by_domain[distinct_type]:
		for keyword in entries_by_domain[distinct_type][keyword_category_set]:
			for domain in contextualised_set_summarised[keyword_category_set][keyword]["frq"]:
				for entry in entries_by_domain[distinct_type][keyword_category_set][keyword][domain]:
					if (not keyword in keyword_domain_demographic_breakdowns):
						keyword_domain_demographic_breakdowns[keyword] = dict()
					if (not domain in keyword_domain_demographic_breakdowns[keyword]):
						keyword_domain_demographic_breakdowns[keyword][domain] = list()
					keyword_domain_demographic_breakdowns[keyword][domain].append(entry["activation_code"])
	for keyword in keyword_domain_demographic_breakdowns:
		for domain in keyword_domain_demographic_breakdowns[keyword]:
			keyword_domain_demographic_breakdowns[keyword][domain] = participant_breakdown(participant_breakdown_filtered(
																							list(set(keyword_domain_demographic_breakdowns[keyword][domain]))))
	with open(f"keyword_domain_demographic_breakdowns.json", "w") as f:
		f.write(json.dumps(keyword_domain_demographic_breakdowns, indent=3))
		f.close()

'''
	This function evaluates how different, by distribution of frequency, each 'keyword-domain-demographic' breakdown is (within
	the 'keyword_domain_demographic_breakdowns.json' file), in comparison to the overall participant cohort statistics.

	This is of relevance, as we are interested in determining whether any domains reported significantly different observations of
	certain domains, as opposed to similar demographic breakdowns to the overall cohort.
'''
def generate_percentage_differences_between_kddb_and_participant_cohort_statistics():
	# Weightings to percentages -> percentage to percentage comparison (subset to entire cohort)
	keyword_domain_demographic_breakdowns = json.loads(open(f"keyword_domain_demographic_breakdowns.json").read())
	differences = dict()
	for keyword in keyword_domain_demographic_breakdowns:
		differences[keyword] = dict()
		for domain in keyword_domain_demographic_breakdowns[keyword]:
			differences[keyword][domain] = dict()
			for demographic_characteristic in keyword_domain_demographic_breakdowns[keyword][domain]:
				total = sum(keyword_domain_demographic_breakdowns[keyword][domain][demographic_characteristic].values())
				if (total > 30):
					differences[keyword][domain][demographic_characteristic] = {discrete_value:
						((keyword_domain_demographic_breakdowns[keyword][domain][demographic_characteristic][discrete_value]/
						total)
							- (participating_cohort_statistics[demographic_characteristic][discrete_value]/
								sum(participating_cohort_statistics[demographic_characteristic].values())))
								for discrete_value in keyword_domain_demographic_breakdowns[keyword][domain][demographic_characteristic]}
				else:
					print(keyword, domain, demographic_characteristic)
	# All keywords had substantial enough frequencies of donations to produce breakdowns, except the 
	# keywords 'Tokyo 2021' and 'Tokyo Olympics', which only had a small portion of data donations for this time window
	del differences["Tokyo 2021"] 
	del differences["Tokyo Olympics"]
	with open(f"differences.json", "w") as f:
		f.write(json.dumps(differences, indent=3))
		f.close()

'''
	This function generates a data-structure that can be graphed, to represent the differences produced from the aforementioned function.
'''
def generate_graphable_differences():
	# DIfferences to graphed values
	differences = json.loads(open(f"differences.json").read())
	graphed_values = dict()
	for keyword in differences:
		for domain in differences[keyword]:
			for demographic_characteristic in differences[keyword][domain]:
				if (demographic_characteristic not in graphed_values):
					graphed_values[demographic_characteristic] = dict()
				for discrete_value in differences[keyword][domain][demographic_characteristic]:
					if (discrete_value not in graphed_values[demographic_characteristic]):
						graphed_values[demographic_characteristic][discrete_value] = list()
					graphed_values[demographic_characteristic][discrete_value].append(differences[keyword][domain][demographic_characteristic][discrete_value])
	with open(f"graphed_values.json", "w") as f:
		f.write(json.dumps(graphed_values, indent=3))
		f.close()

'''
	This function generates the set of contextualised set of observations, for the various observed domains, for all keywords,
	segmented by the various demographic characteristics that were considered.
'''
def generate_contextualised_set():
	# For a given stat
	# For a given keyword
	top_sources_n = 100
	top_sources_dict = dict()
	participants = {x["activation_code"]:x for x in json.loads(open("participating_cohort.json").read())}
	for k_type in ["k1", "k2"]:
		contextualised_set = dict()
		declined_participants = list()
		ii = 0
		for this_keyword in entries_by_domain_summarised["nondistinct"][k_type]:
			top_sources_dict[this_keyword] = dict()
			top_sources = [{"domain" : k, "avg" : v["avg"], "frq" : v["frq"]} for k,v in entries_by_domain_summarised["nondistinct"][k_type][this_keyword].items()]
			top_sources_dict[this_keyword]["frq"] = sorted(top_sources, key=lambda d: d['frq'], reverse=True)#[:top_sources_n]
			top_sources_dict[this_keyword]["avg"] = sorted(top_sources, key=lambda d: d['avg'])#[:top_sources_n]
		# For each entry
		for entry in [json.loads(x) for x in open(f"output.jsond.temporal.{k_type}.p4").read().split("\n") if (len(x) > 0)]:
			ii += 1
			if (ii % 10000 == 0):
				print(ii)
			# Determine the relevant participant
			if (entry["activation_code"] in participants):
				this_participant = participants[entry["activation_code"]]
				# For each keyword
				for this_keyword in entries_by_domain_summarised["nondistinct"][k_type]:
					# For each stat
					for this_stat in ["frq", "avg"]:
						# For each of the top sources (removing suffixes on domain names)
						for i in range(len(top_sources_dict[this_keyword][this_stat])):
							this_source = top_sources_dict[this_keyword][this_stat][i]["domain"]
							# If the source is within the entry
							if (this_source == entry["source_url_fld"] + f" ({entry['country']})"):
								# For each of the relevant demographic characteristics
								for this_dc in ["gender", "age", "postcode", "income_bracket", "level_education", "employment_status", "party_preference"]:
									this_dc_v = this_participant[this_dc]
									if (this_dc == "postcode"):
										this_dc_v = postcode_to_state(this_dc_v)
									# Prepopulate the contextualised set (if unpopulated)
									if (not this_keyword in contextualised_set):
										contextualised_set[this_keyword] = dict()
									if (not this_stat in contextualised_set[this_keyword]):
										contextualised_set[this_keyword][this_stat] = dict()
									if (not this_source in contextualised_set[this_keyword][this_stat]):
										contextualised_set[this_keyword][this_stat][this_source] = dict()
									if (not this_dc in contextualised_set[this_keyword][this_stat][this_source]):
										contextualised_set[this_keyword][this_stat][this_source][this_dc] = dict()
									if (not this_dc_v in contextualised_set[this_keyword][this_stat][this_source][this_dc]):
										contextualised_set[this_keyword][this_stat][this_source][this_dc][this_dc_v] = list()
									# Add the list index to the contextualised set
									contextualised_set[this_keyword][this_stat][this_source][this_dc][this_dc_v].append(entry["list_index"])
			else:
				if (not entry["activation_code"] in declined_participants):
					declined_participants.append(entry["activation_code"])
		with open(f"contextualised_set.{k_type}.json", "w") as f:
			f.write(json.dumps(contextualised_set, indent=3))
			f.close()
#generate_contextualised_set()

graphed_values = json.loads(open("graphed_values.json").read())
ordered_headings = json.loads(open("ordered_headings.json").read())
headings_formalised = json.loads(open("headings_formalised.json").read())

'''
	For a demographic characteristic (of either 'gender', 'age', 'postcode' (which resolves to state), 'level_education', 'employment_status',
	'party_preference', 'level_education', or 'income_bracket'), this function plots the overall trend in differences that each of the observed publishers
	have in terms of the number of participants from the nominated demographic that reported observing the publisher, in comparison to the overall dataset.

	While the visualisation does not single out any individual publisher, it still tallies the difference (as a percentage) that said publisher's distribution
	of participants differ from the overall cohort of participants (with respect to the demographic characteristic in question).

	This is of relevance for determining whether or not certain demographic characteristics were inclined to observe certain news sources more than others.
	Pronounced effects would warrant a significant percentage of difference to the entire participant cohort.
'''

def plot_overall_differences(demographic_characteristic, show=False):
	discrete_values = graphed_values[demographic_characteristic].keys()
	MAX_PER_LINE = 3
	x_size = MAX_PER_LINE
	y_size = math.ceil(len(discrete_values)/MAX_PER_LINE)
	fig, ax = plt.subplots(y_size, x_size)
	fig.set_size_inches(18.5, 4*y_size, forward=True)
	fig.subplots_adjust(hspace=0.3)
	for ii in range(x_size*y_size):
		xx = (ii % MAX_PER_LINE)
		yy = math.floor(ii/x_size)
		if (ii < len(list(discrete_values))):
			v = ordered_headings[demographic_characteristic][ii]
			this_mean = (np.mean(graphed_values[demographic_characteristic][v]))
			this_std = (np.std(graphed_values[demographic_characteristic][v]))
			ax[yy,xx].text(0.95, 0.9, 'Mean: %2.3f' % (this_mean), size=9, color='black',
						horizontalalignment='right', verticalalignment='top', transform=ax[yy,xx].transAxes)
			ax[yy,xx].text(0.95, 0.825, 'Std: %2.3f' % (this_std), size=9, color='black',
						horizontalalignment='right', verticalalignment='top', transform=ax[yy,xx].transAxes)
			bins = int(np.std(graphed_values[demographic_characteristic][v])*2500)
			sns.histplot(graphed_values[demographic_characteristic][v], bins=bins, ax=ax[yy,xx], stat='density')
			sns.kdeplot(graphed_values[demographic_characteristic][v], ax=ax[yy,xx], fill=True)
			ax[yy,xx].set_title(v)
			ax[yy,xx].set_ylabel(str())
			ax[yy,xx].set_xlim(-0.1,+0.1)
			ax[yy,xx].set_xticks([-0.1,-0.05,0,0.05,0.1])
			ax[yy,xx].set_xticklabels(["-10%","-5%","0","+5%","+10%"])
			ax[yy,xx].set_xlabel("Difference (%)")
			ax[yy,xx].set_ylabel("Frequency")
		else:
			ax[yy,xx].axis('off')
	fig.suptitle(f"Overall Differences In Observation Of Cumulative Publishers For Demographic Characteristic '{headings_formalised[demographic_characteristic]}'",fontsize=18)
	if (show):
		plt.show()

# Load in the keyword-domain-demographic breakdowns
keyword_domain_demographic_breakdowns = json.loads(open("keyword_domain_demographic_breakdowns.json").read())
# Load in the formalised headings
headings_formalised = json.loads(open("headings_formalised.json").read())

def plot_avg_list_indices(this_keyword, this_domain, this_stat, this_dc, ktype="k1", show=False):
	cset = contextualised_set_k1 if (ktype == "k1") else contextualised_set_k2
	fig, ax = plt.subplots(1, 2)
	fig.set_size_inches(13, 4, forward=True)
	this_labels = ordered_headings[this_dc]
	width = 0.3
	nrange = np.arange(len(this_labels))
	ax[0].bar([x for x in this_labels if (x in cset[this_keyword][this_stat][this_domain][this_dc])], 
		height=[np.mean(cset[this_keyword][this_stat][this_domain][this_dc][x]) for x in this_labels
			   if (x in cset[this_keyword][this_stat][this_domain][this_dc])])
	ax[0].set_xlabel(headings_formalised[this_dc])
	ax[0].set_ylabel("Avg. List Index")
	ax[0].set_xticks(nrange, this_labels,rotation=90)
	xxx = [0 if (not x in cset[this_keyword][this_stat][this_domain][this_dc]) else len(cset[this_keyword][this_stat][this_domain][this_dc][x]) for x in this_labels]
	try:
		sum_all = sum(list(keyword_domain_demographic_breakdowns[this_keyword][this_domain][this_dc].values()))
	except:
		sum_all = 0
	yyy = [(0 if ((not this_domain in keyword_domain_demographic_breakdowns[this_keyword]) or (not x in keyword_domain_demographic_breakdowns[this_keyword][this_domain][this_dc])) 
			   else keyword_domain_demographic_breakdowns[this_keyword][this_domain][this_dc][x]/sum_all*sum(xxx)) for x in this_labels]
	ax[1].bar(nrange, xxx, width, label="Actual Frequency")
	ax[1].bar(nrange + width, yyy, width, label="Predicted Frequency From Entire Cohort")
	ax[1].set_xticks(nrange + width / 2, this_labels, rotation=90)
	ax[1].set_xlabel(headings_formalised[this_dc])
	ax[1].set_ylabel("Frq. Of Observations")
	ax[1].legend()
	fig.suptitle(f"Avg. List Indices And Frequencies Of Observation For Publisher '{this_domain}' On Keyword '{this_keyword}'\n- Demographic Characteristic: {headings_formalised[this_dc]} -")
	if (show):
		plt.show()

'''
	This function plots the data contained in the contextualised set that corresponds to the first set of keywords
'''
def plot_contextualised_set_k1():
	for this_keyword in contextualised_set_k1:
		for this_stat in ["frq"]:
			for this_domain in contextualised_set_k1[this_keyword][this_stat]:
				for this_dc in contextualised_set_k1[this_keyword][this_stat][this_domain]:
					plot_avg_list_indices(this_keyword, this_domain, this_stat, this_dc, show=GLOBAL_SHOW)

'''
	This function plots the data contained in the contextualised set that corresponds to the second set of keywords
'''
def plot_contextualised_set_k2():
	for this_keyword in contextualised_set_k2:
		for this_stat in ["frq"]:
			for this_domain in contextualised_set_k2[this_keyword][this_stat]:
				for this_dc in contextualised_set_k2[this_keyword][this_stat][this_domain]:
					plot_avg_list_indices(this_keyword, this_domain, this_stat, this_dc, ktype="k2", show=GLOBAL_SHOW)
values_mapped = {
		"age" : ['18 - 24', '25 - 34', '35 - 44', '45 - 54', '55 - 64', '65 - 74', '75 and over', 'Prefer not to say'],
		"gender" : ['Male', 'Female', 'Other', 'Prefer not to say'],
		"employment_status" : ['Employed full-time', 'Employed part-time', 'Unemployed and looking for work', 'Unemployed and not looking for work', 'Retired', 'Prefer not to say'],
		"state" : ['VIC', 'NSW', 'TAS', 'QLD', 'SA', 'ACT', 'WA', 'NT'],
		"postcode" : ['VIC', 'NSW', 'TAS', 'QLD', 'SA', 'ACT', 'WA', 'NT'],
		"income_bracket" : ['$1 - $15,599', '$15,600 - $20,799', '$20,800 - $25,999', '$26,000 - $33,799', '$33,800 - $41,599', '$41,600 - $51,999', '$52,000 - $64,999', '$65,000 - $77,999', '$78,000 - $90,999', '$91,000 - $103,999', '$104,000 - $155,999','$156,000 or more', 'Prefer not to say'],
		"party_preference" : ['Labor', 'Greens', 'Liberal', 'National', 'One Nation', 'Other', 'None', 'Prefer not to say'],
		"level_education" : ['Less than year 12 or equivalent', 'Year 12 or equivalent', 'Bachelor degree level', 'Postgraduate degree level', 'Prefer not to say']
	}

distributions_titles_mapped_ase = {
		"state" : "State Of Residence",
		"postcode" : "State Of Residence",
		"gender" : "Gender",
		"employment_status" : "Employment Status",
		"age" : "Age (excluding under 18)",
		"level_education" : "Education Level",
		"income_bracket" : "Income Bracket",
		"party_preference" : "Political Party Preference"
	}

def plot_average_search_rankings_vs_frequencies(inserted_title, this_demographic_characteristic, ordered_values, avg_list_indices, frequencies, pvals, desired_pvalue=0.05, show=False):
	try:
		if ((show) or (not os.path.exists(os.path.join(os.getcwd(), "avg_search_rankings_vs_frequencies", f"{inserted_title} - Avg. Search Rankings vs. Frequencies.png")))):
			desired_pvalue = 0.05
			fig, ax = plt.subplots(1,2)
			fig.set_size_inches(10,3.5) 
			pvals_applied = {k:(None if (not k in pvals) else pvals[k]) for k in ordered_values}
			avg_list_indices_applied = {k:(0 if (not k in avg_list_indices) else avg_list_indices[k]) for k in ordered_values}
			bar1 = ax[0].barh(ordered_values, width=[avg_list_indices_applied[x] 
													 for x in ordered_values], align='center', height=0.4, color="#557a9e")
			max_width_bar1 = max([rect.get_width() for rect in bar1])
			ii = 0
			for rect in bar1:
				this_color = "black"
				this_pval = [pvals_applied[x] for x in ordered_values][ii]
				if ((this_pval is None) or (this_pval > desired_pvalue)):
					this_color = "#ad4953"
					rect.set_color("#ad4953")
				width = rect.get_width() + (max_width_bar1*0.075)
				ax[0].text(width, rect.get_y() + (rect.get_height() / 2.0), 
						   f'{rect.get_width()+1:.2f}', ha='center', va='center', fontsize=10, color=this_color)
				ii += 1
			ax[0].set_xlim(0, max_width_bar1*1.3)
			ax[0].set_xticks([x/1000 for x in list(range(0,round(max_width_bar1*1.3*1000) + round(max_width_bar1*1.3/10*1000), round(max_width_bar1*1.3/10*1000)))])
			ax[0].set_xticklabels([f"{(x + 1):.2f}" for x in [y/1000 for y in list(range(0,round(max_width_bar1*1.3*1000) + round(max_width_bar1*1.3/10*1000), round(max_width_bar1*1.3/10*1000)))]])
			ax[0].invert_yaxis()
			ax[0].set_ylabel(distributions_titles_mapped_ase[this_demographic_characteristic])
			ax[0].set_xlabel('Avg. Search Ranking')
			ax[0].set_yticklabels([x.replace("$", "\\$") for x in ordered_values])
			ax[0].grid(color="#f5f5f5")
			ax[0].set_axisbelow(True)
			frequencies_applied = {k:(0 if (not k in frequencies) else frequencies[k]) for k in ordered_values}
			bar2 = ax[1].barh(ordered_values, width=[frequencies_applied[x] 
													 for x in ordered_values], align='center', height=0.4, color="#557a9e")
			max_width_bar2 = max([rect.get_width() for rect in bar2])
			ii = 0
			for rect in bar2:
				width = rect.get_width() + (max_width_bar2*0.05)
				this_color = "black"
				this_pval = [pvals_applied[x] for x in ordered_values][ii]
				if ((this_pval is None) or (this_pval > desired_pvalue)):
					this_color = "#ad4953"
					rect.set_color("#ad4953")
				ax[1].text(width, rect.get_y() + (rect.get_height() / 2.0)*0.9, 
						   "Frq: "+str(rect.get_width()), ha='left', va='bottom', fontsize=8, color=this_color)
				ax[1].text(width, rect.get_y() + (rect.get_height() / 2.0)*1.1, 
						   "p-value: "+("Inconclusive" if (this_pval is None) else (str(f"{this_pval:.03g}"))), ha='left', va='top', fontsize=8, color=this_color)
				ii += 1
			ax[1].set_xlim(0, max_width_bar2*1.4)
			ax[1].set_yticklabels(list())
			#ax.set_yticklabels(people)
			ax[1].invert_yaxis()  # labels read top-to-bottom
			ax[1].set_xlabel('Frequency')
			ax[1].grid(color="#f5f5f5")
			ax[1].set_axisbelow(True)
			#plt.suptitle(f"{inserted_title} - Avg. Search Rankings vs. Frequencies")
			plt.subplots_adjust(wspace=0.05,bottom=0.15, left=0.15, right=0.95)
			plt.gcf().subplots_adjust(wspace=0.05,bottom=0.15, left=0.15, right=0.95)
			plt.tight_layout()
			if (not show):
				plt.savefig(os.path.join(os.getcwd(), "avg_search_rankings_vs_frequencies", f"{inserted_title} - Avg. Search Rankings vs. Frequencies.png"), dpi=200)
				plt.clf()
				plt.close()
			else:
				plt.show()
		else:
			print("Skipping: ", inserted_title)
	except:
		print(traceback.format_exc())
		if (not show):
			print("Failed on: ", inserted_title)

'''
	This function generates the summarisation of the contextualised set, which provides details on p values and
	frequencies for visualisation.
'''
def generate_contextualised_set_summarised():
	contextualised_set_summarised = dict()
	for ktype in contextualised_set:
		for this_keyword in contextualised_set[ktype]:
			for this_stat in ["frq", "avg"]:
				for this_domain in contextualised_set[ktype][this_keyword][this_stat]:
					for this_dc in contextualised_set[ktype][this_keyword][this_stat][this_domain]:
						for v in contextualised_set[ktype][this_keyword][this_stat][this_domain][this_dc]:
							this_array = contextualised_set[ktype][this_keyword][this_stat][this_domain][this_dc][v]
							this_frq = len(this_array)
							this_avg_search_result_ranking = np.mean(this_array)
							ntest_pvalue = None
							ntest_statistic = None
							try:
								test_result = scipy.stats.normaltest(this_array)
								ntest_pvalue = test_result.pvalue
								ntest_statistic = test_result.statistic
							except:
								pass

							if (not ktype in contextualised_set_summarised):
								contextualised_set_summarised[ktype] = dict()
							if (not this_keyword in contextualised_set_summarised[ktype]):
								contextualised_set_summarised[ktype][this_keyword] = dict()
							if (not this_stat in contextualised_set_summarised[ktype][this_keyword]):
								contextualised_set_summarised[ktype][this_keyword][this_stat] = dict()
							if (not this_domain in contextualised_set_summarised[ktype][this_keyword][this_stat]):
								contextualised_set_summarised[ktype][this_keyword][this_stat][this_domain] = dict()
							if (not this_dc in contextualised_set_summarised[ktype][this_keyword][this_stat][this_domain]):
								contextualised_set_summarised[ktype][this_keyword][this_stat][this_domain][this_dc] = dict()
							contextualised_set_summarised[ktype][this_keyword][this_stat][this_domain][this_dc][v] = {
									"frq" : this_frq,
									"avg" : this_avg_search_result_ranking,
									"ntest_pvalue" : ntest_pvalue,
									"ntest_statistic" : ntest_statistic
								}
	with open("contextualised_set_summarised.json", "w") as f:
		f.write(json.dumps(contextualised_set_summarised,indent=3))
		f.close()

def rearrange_contextualised_set_by_domain(contextualised_set_summarised=contextualised_set_summarised):
	domain_dict = dict()
	for ktype in contextualised_set_summarised:
		for keyword in contextualised_set_summarised[ktype]:
			for stat in contextualised_set_summarised[ktype][keyword]:
				for domain in contextualised_set_summarised[ktype][keyword][stat]:
					for dc in contextualised_set_summarised[ktype][keyword][stat][domain]:
						for v in contextualised_set_summarised[ktype][keyword][stat][domain][dc]:
							if (not ktype in domain_dict):
								domain_dict[ktype] = dict()
							if (not keyword in domain_dict[ktype]):
								domain_dict[ktype][keyword] = dict()
							if (not stat in domain_dict[ktype][keyword]):
								domain_dict[ktype][keyword][stat] = dict()
							if (not domain in domain_dict[ktype][keyword][stat]):
								domain_dict[ktype][keyword][stat][domain] = dict()
							if (not dc in domain_dict[ktype][keyword][stat][domain]):
								domain_dict[ktype][keyword][stat][domain][dc] = dict()
							if (not "avg_list_indices" in domain_dict[ktype][keyword][stat][domain][dc]):
								domain_dict[ktype][keyword][stat][domain][dc]["avg_list_indices"] = dict()
							if (not "frequencies" in domain_dict[ktype][keyword][stat][domain][dc]):
								domain_dict[ktype][keyword][stat][domain][dc]["frequencies"] = dict()
							if (not "pvals" in domain_dict[ktype][keyword][stat][domain][dc]):
								domain_dict[ktype][keyword][stat][domain][dc]["pvals"] = dict()
							if (not "pvals_stats" in domain_dict[ktype][keyword][stat][domain][dc]):
								domain_dict[ktype][keyword][stat][domain][dc]["pvals_stats"] = dict()
							domain_dict[ktype][keyword][stat][domain][dc]["pvals"][v] = contextualised_set_summarised[ktype][keyword][stat][domain][dc][v]["ntest_pvalue"]
							domain_dict[ktype][keyword][stat][domain][dc]["pvals_stats"][v] = contextualised_set_summarised[ktype][keyword][stat][domain][dc][v]["ntest_statistic"]
							domain_dict[ktype][keyword][stat][domain][dc]["frequencies"][v] = contextualised_set_summarised[ktype][keyword][stat][domain][dc][v]["frq"]
							domain_dict[ktype][keyword][stat][domain][dc]["avg_list_indices"][v] = contextualised_set_summarised[ktype][keyword][stat][domain][dc][v]["avg"]
	with open("contextualised_set_summarised_by_domain.json", "w") as f:
		f.write(json.dumps(domain_dict,indent=3))
		f.close()

contextualised_set_summarised_by_domain = json.loads(open("contextualised_set_summarised_by_domain.json").read())
#rearrange_contextualised_set_by_domain(contextualised_set_summarised)
try:
	os.mkdir(os.path.join(os.getcwd(), "avg_search_rankings_vs_frequencies"))
except:
	pass
try:
	os.mkdir(os.path.join(os.getcwd(), "representation_distributions"))
except:
	pass

def generate_all_plot_average_search_rankings_vs_frequencies():
	for this_stat in ["frq", "avg"]:
		this_stat_title = "Most Frequent News Source" if (this_stat == "frq") else "Highest Ranked News Source"
		for ktype in contextualised_set_summarised_by_domain:
			for this_keyword in contextualised_set_summarised_by_domain[ktype]:
				alt_ktype = "k1" if (ktype == "k2") else "k2"
				if (not this_keyword in contextualised_set_summarised_by_domain[alt_ktype]):
						for this_domain_name in contextualised_set_summarised_by_domain[ktype][this_keyword][this_stat]:
							for this_demographic_characteristic in contextualised_set_summarised_by_domain[ktype][this_keyword][this_stat][this_domain_name]:
								this_domain_details = contextualised_set_summarised_by_domain[ktype][this_keyword][this_stat][this_domain_name][this_demographic_characteristic]
								plot_average_search_rankings_vs_frequencies(
									f"Keyword '{this_keyword}' - " 
										+ f"{this_stat_title} '{this_domain_name}' - "
										+ f"{distributions_titles_mapped_ase[this_demographic_characteristic]}", 
									this_demographic_characteristic, 
									values_mapped[this_demographic_characteristic], 
									this_domain_details["avg_list_indices"], 
									this_domain_details["frequencies"], 
									this_domain_details["pvals"], desired_pvalue=0.05, show=GLOBAL_SHOW)

def compact_plot_average_search_rankings_vs_frequencies(ktype, this_keyword, this_demographic_characteristic, this_domain_name, this_stat, show=GLOBAL_SHOW):
	this_domain_details = contextualised_set_summarised_by_domain[ktype][this_keyword][this_stat][this_domain_name][this_demographic_characteristic]
	this_stat_title = "Most Frequent News Source" if (this_stat == "frq") else "Highest Ranked News Source"
	plot_average_search_rankings_vs_frequencies(
		f"Keyword '{this_keyword}' - " + f"{this_stat_title} '{this_domain_name}' - "
		+ f"{distributions_titles_mapped_ase[this_demographic_characteristic]}", 
		this_demographic_characteristic, values_mapped[this_demographic_characteristic], 
		this_domain_details["avg_list_indices"], this_domain_details["frequencies"], 
		this_domain_details["pvals"], desired_pvalue=0.05, show=show)

# maps abs to ase (if possible)
abs_dc_responses_map_to = {
	"state" : {
		"QLD" : "QLD", 
		"NSW" : "NSW", 
		"VIC" : "VIC", 
		"ACT" : "ACT", 
		"NT" : "NT", 
		"WA" : "WA", 
		"TAS" : "TAS", 
		"SA" : "SA"
	},
	"gender" : {
		"Male" : "Male", 
		"Female" : "Female"
	},
	"employment_status" : {
		"Employed" : "Employed", 
		"Retired" : "Retired", 
		"Unemployed" : "Unemployed", 
		"Undescribed" : "Prefer not to say"
	},
	"party_preference" : {
		"Australian Labor Party" : "Labor",
		"Australian Greens" : "Greens", 
		"Liberal National Party" : "Liberal", 
		"Undescribed" : "Undescribed"
	},
	"income_bracket" : {
		"$1 - $51,999" : "$1 - $51,999",
		"$52,000 - $103,999" : "$52,000 - $103,999", 
		"$104,000 - $155,999" : "$104,000 - $155,999", 
		"Undescribed" : "Undescribed"
	},
	"level_education" : {
		"Secondary or less" : "Secondary or less", 
		"Tertiary" : "Tertiary", 
		"Undescribed" : "Prefer not to say"
	},
	"age" : {
		"18 - 34" : "18 - 34", 
		"35 - 64" : "35 - 64", 
		"65 and over" : "65 and over"
	}
}

# aggregate ase responses to abs
abs_dc_responses_makeup = {
	"employment_status" : {
		"Employed" : ["Employed full-time", "Employed part-time"], 
		"Unemployed" : ["Unemployed and looking for work", "Unemployed and not looking for work"]
	},
	"party_preference" : {
		"Undescribed" : ["National", "One Nation", "Other", "None", "Prefer not to say"]
	},
	"income_bracket" : {
		"$1 - $51,999" : ["$1 - $15,599", "$15,600 - $20,799", "$20,800 - $25,999", "$26,000 - $33,799", "$33,800 - $41,599", "$41,600 - $51,999"],
		"$52,000 - $103,999" : ["$52,000 - $64,999", "$65,000 - $77,999", "$78,000 - $90,999", "$91,000 - $103,999"],
		"Undescribed" : ["$156,000 or more", "Prefer not to say"]
	},
	"level_education" : {
		"Secondary or less" : ["Less than year 12 or equivalent", "Year 12 or equivalent"],
		"Tertiary" : ["Bachelor degree level", "Postgraduate degree level"]
	},
	"age" : {
		"18 - 34" : ["18 - 24", "25 - 34"],
		"35 - 64" : ["35 - 44", "45 - 54", "55 - 64"],
		"65 and over" : ["65 - 74", "75 and over"]
	}
}


# orders headings (integrating ase and abs into one)
abs_ase_unified_ordered_headings = {
	"state" : [
		"QLD", "NSW", "VIC", "ACT", "NT", "WA", "TAS", "SA"
	],
	"gender" : [
		"Male", "Female", "Other", "Prefer not to say"
	],
	"employment_status" : [
		"Employed full-time", "Employed part-time", "Employed", "Retired", "Unemployed and looking for work", "Unemployed and not looking for work", 
		"Unemployed", "Prefer not to say"
	],
	"party_preference" : [
		"Labor", "Greens", "Liberal", "National", "One Nation", "Other", "None", "Prefer not to say", "Undescribed"
	],
	"income_bracket" : [
		"$1 - $15,599", "$15,600 - $20,799", "$20,800 - $25,999", "$26,000 - $33,799", "$33,800 - $41,599", "$41,600 - $51,999", "$1 - $51,999",
		"$52,000 - $64,999", "$65,000 - $77,999", "$78,000 - $90,999", "$91,000 - $103,999", "$52,000 - $103,999",
		"$156,000 or more", "$104,000 - $155,999", "Prefer not to say", "Undescribed"
	],
	"level_education" : [
		"Less than year 12 or equivalent", "Year 12 or equivalent", "Secondary or less", "Bachelor degree level", 
		"Postgraduate degree level", "Tertiary", "Prefer not to say"
	],
	"age" : [
		"18 - 24", "25 - 34", "18 - 34", "35 - 44", "45 - 54", "55 - 64", "35 - 64", "65 - 74", "75 and over", "65 and over", "Prefer not to say"
	]
}

abs_distributions = json.loads(open("abs_distributions.json").read())

ase_headings_to_abs = {
	"state" : "STATE_OF_RESIDENCE",
	"gender" : "GENDER",
	"employment_status" : "EMPLOYMENT_STATUS",
	"age" : "AGE",
	"level_education" : "EDUCATION_LEVEL",
	"income_bracket" : "INCOME_BRACKET",
	"party_preference" : "POLITICAL_PARTY_PREFERENCE"
}

representations_compared = json.loads(open("representations_compared.json").read())

#contextualised_set_uncut_nondistinct
def generate_representations_compared():
	representations_compared = dict()
	for ktype in contextualised_set_summarised:
		for this_keyword in contextualised_set_summarised[ktype]:
			for this_domain in contextualised_set_summarised[ktype][this_keyword]["frq"]:
				for demographic_characteristic in contextualised_set_summarised[ktype][this_keyword]["frq"][this_domain]:
					applied_demographic_characteristic = "state" if (demographic_characteristic == "postcode") else demographic_characteristic
					# Compile the overall cohort's demographic distributions as percentages
					all_domains_cohorts = list()
					for a_domain in contextualised_set_summarised[ktype][this_keyword]["frq"]:
						if (demographic_characteristic in contextualised_set_summarised[ktype][this_keyword]["frq"][a_domain]):
							some_cohort_ase = contextualised_set_summarised[ktype][this_keyword]["frq"][a_domain][demographic_characteristic]
							some_cohort_ase = {x:some_cohort_ase[x]["frq"] for x in some_cohort_ase}
							#ipdb.set_trace()
							some_cohort_ase_pct = {k:(some_cohort_ase[k]/sum(some_cohort_ase.values())) for k in some_cohort_ase}
							some_cohort_ase_pct = {k:float() if (not k in some_cohort_ase_pct) else some_cohort_ase_pct[k] for k in abs_ase_unified_ordered_headings[applied_demographic_characteristic]}
							all_domains_cohorts.append(some_cohort_ase_pct)
					cohort_ase_pct = {k:sum([c[k] for c in all_domains_cohorts])/len(all_domains_cohorts) for k in all_domains_cohorts[0]}
					# Compile the non-distinct observations as percentages
					cohort_relative = contextualised_set_summarised[ktype][this_keyword]["frq"][this_domain][demographic_characteristic]
					cohort_relative = {x:cohort_relative[x]["frq"] for x in cohort_relative}
					cohort_relative_pct = {k:(cohort_relative[k]/sum(cohort_relative.values())) for k in cohort_relative}
					cohort_relative_pct = {k:float() if (not k in cohort_relative_pct) else cohort_relative_pct[k] for k in abs_ase_unified_ordered_headings[applied_demographic_characteristic]}
					# Attempt to fill in responses for ABS (where possible)
					if (demographic_characteristic in abs_dc_responses_makeup):
						for abs_response in abs_dc_responses_makeup[demographic_characteristic]:
							cohort_relative_pct[abs_response] = sum([cohort_relative_pct[x] 
								for x in abs_dc_responses_makeup[demographic_characteristic][abs_response]])
							cohort_ase_pct[abs_response] = sum([cohort_ase_pct[x] 
								for x in abs_dc_responses_makeup[demographic_characteristic][abs_response]])
					# Compile the ABS cohort's demographic distributions as percentages
					cohort_abs_pct = {
						(k if (not demographic_characteristic in abs_dc_responses_map_to) else abs_dc_responses_map_to[demographic_characteristic][k]):v 
												for k,v in abs_distributions[ase_headings_to_abs[applied_demographic_characteristic]].items()}
					cohort_abs_pct = {k:float() if (not k in cohort_abs_pct) else cohort_abs_pct[k] for k in abs_ase_unified_ordered_headings[applied_demographic_characteristic]}
					
					if (not ktype in representations_compared):
						representations_compared[ktype] = dict()
					if (not this_keyword in representations_compared[ktype]):
						representations_compared[ktype][this_keyword] = dict()
					if (not this_domain in representations_compared[ktype][this_keyword]):
						representations_compared[ktype][this_keyword][this_domain] = dict()

					representations_compared[ktype][this_keyword][this_domain][applied_demographic_characteristic] = {
							"cohort_relative_pct" : cohort_relative_pct,
							"cohort_ase_pct" : cohort_ase_pct,
							"cohort_abs_pct" : cohort_abs_pct
						}
	with open("representations_compared.json", "w") as f:
		f.write(json.dumps(representations_compared,indent=3))
		f.close()

def plot_representation_distributions(ktype, this_keyword, this_demographic_characteristic_arg, this_domain, show=False, normalisation=False):
	this_demographic_characteristic = "state" if (this_demographic_characteristic_arg == "postcode") else this_demographic_characteristic_arg
	try:
		if ((this_domain in representations_compared[ktype][this_keyword]) and ((show) or (not os.path.exists(os.path.join(os.getcwd(), "representation_distributions", f"Representation Of {distributions_titles_mapped_ase[this_demographic_characteristic]} - Keyword '{this_keyword}' - News Source '{this_domain}'.png"))))):
			comparison_data = representations_compared[ktype][this_keyword][this_domain][this_demographic_characteristic]
			responses = abs_ase_unified_ordered_headings[this_demographic_characteristic]
			fig, ax = plt.subplots(1,1)
			fig.set_size_inches(10,3.5) 
			ind = np.arange(len(responses))
			width = 0.2	   
			l3 = list()
			if (normalisation):
				l1 = list(comparison_data["cohort_relative_pct"].values())
				l2 = list(comparison_data["cohort_ase_pct"].values())
				for i in range(len(l1)):
					l3.append(l1[i]/l2[i])
				ax.bar(ind, l3, width, label='Participants Who Saw News Source (Normalised)', color="#47a67d")
			else:
				ax.bar(ind, comparison_data["cohort_relative_pct"].values() , width, 
							   label='Participants Who Saw News Source', color="#47a67d")
				ax.bar(ind + width, comparison_data["cohort_ase_pct"].values(), width, 
							   label='Participants Of Australian Search Experience', color="#4799a6")
				ax.bar(ind + (width*2), comparison_data["cohort_abs_pct"].values(), width, 
							   label='Participants Of Australian Census 2021', color="#57559e")
			ax.set_xlabel(distributions_titles_mapped_ase[this_demographic_characteristic])
			ax.set_ylabel('Percentage (%)')
			#ax.set_title(f"Representation Of {distributions_titles_mapped_ase[this_demographic_characteristic]} - Keyword '{this_keyword}' - News Source '{this_domain}'")
			if (normalisation):
				max_pct = max(l3)
			else:
				max_pct = max(list(comparison_data["cohort_relative_pct"].values())
					+ list(comparison_data["cohort_ase_pct"].values())
					+ list(comparison_data["cohort_abs_pct"].values()))
			ax.set_ylim(0, max_pct*1.3)
			ax.set_yticks([x/1000 for x in list(range(0,round(max_pct*1.3*1000) 
							+ round(max_pct*1.3/10*1000), round(max_pct*1.3/10*1000)))])
			ax.set_yticklabels([f"{round(x*100)}%" for x in [y/1000 for y in list(range(0,round(max_pct*1.3*1000) 
														+ round(max_pct*1.3/10*1000), round(max_pct*1.3/10*1000)))]])
			ax.set_xticks(ind + (width*2) / 2, responses)
			ax.set_xticklabels([x.replace("$", "\\$") for x in responses], rotation=90)
			ax.legend(loc='best')
			ax.grid(color="#f5f5f5")
			ax.set_axisbelow(True)
			if (not show):
				plt.subplots_adjust(wspace=0.05,bottom=0.15)
				plt.gcf().subplots_adjust(wspace=0.05,bottom=0.15)
				plt.tight_layout()
				plt.savefig(os.path.join(os.getcwd(), "representation_distributions", f"Representation Of {distributions_titles_mapped_ase[this_demographic_characteristic]} - Keyword '{this_keyword}' - News Source '{this_domain}'.png"), dpi=200)
				plt.clf()
				plt.close()
			else:
				plt.show()
		else:
			print("Skipping: ",  f"Representation Of {distributions_titles_mapped_ase[this_demographic_characteristic]} - Keyword '{this_keyword}' - News Source '{this_domain}'.png")
	except:
		print(traceback.format_exc())
		if (not show):
			print("Failed on: ", f"Representation Of {distributions_titles_mapped_ase[this_demographic_characteristic]} - Keyword '{this_keyword}' - News Source '{this_domain}'.png")

def generate_all_representation_distributions():
	for this_stat in ["frq", "avg"]:
		for ktype in contextualised_set_summarised_by_domain:
			for this_keyword in contextualised_set_summarised_by_domain[ktype]:
				alt_ktype = "k1" if (ktype == "k2") else "k2"
				if (not this_keyword in contextualised_set_summarised_by_domain[alt_ktype]):
					for this_domain_name in contextualised_set_summarised_by_domain[ktype][this_keyword][this_stat]:
						for this_demographic_characteristic in contextualised_set_summarised_by_domain[ktype][this_keyword][this_stat][this_domain_name]:
							plot_representation_distributions(ktype, this_keyword, this_demographic_characteristic, this_domain_name, show=GLOBAL_SHOW)


if (__name__ == "__main__"):
	#pass
	#generate_contextualised_set()
	#generate_contextualised_set_summarised()
	#generate_contributing_activation_codes()
	#rearrange_contextualised_set_by_domain()
	#generate_representations_compared()

	generate_keyword_domain_demographic_breakdowns()
	generate_percentage_differences_between_kddb_and_participant_cohort_statistics()
	generate_graphable_differences()
	
	#plot_contextualised_set_k1()
	#plot_contextualised_set_k2()
	#generate_all_plot_average_search_rankings_vs_frequencies()
	#compact_plot_average_search_rankings_vs_frequencies()
	#generate_all_representation_distributions()
	#compact_plot_average_search_rankings_vs_frequencies("k1", "Anthony Albanese", "age", "bluemountainsgazette.com.au (AU)", "frq", show=False)
	#plot_representation_distributions("k1", "Barnaby Joyce", "state", "canberratimes.com.au (AU)", show=False)