'''
	This script is responsible for the implementation of the publishers analysis.

	The script is divided into 10 processes that are responsible for retrieving and transforming the Australian Search Experience
	Google News publisher data, as well as visualising the data as bar graphs, treemaps, and pie charts
'''

import re
import os
import sys
import time
import base64
import ipdb
import json
import math
import scipy
import datetime
import itertools
import traceback
import statistics
import matplotlib
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from google.cloud import bigquery
from urllib.parse import urlparse
import matplotlib.colors as colors
from collections import defaultdict
import matplotlib.patches as mpatches

verbose = False

'''
	Simple Google BQ cursor function
'''
def simple_cursor(bq, arg_statement):
	return bq.query(arg_statement).result()

'''
	The first process is responsible for obtaining the Australian Search Experience 'Google News' publisher data
	from an online Google BigQuery dataset.
'''
def process_1():
	os.environ["GOOGLE_APPLICATION_CREDENTIALS"]=os.path.join(os.getcwd(), "creds.json")
	CONSTANT_PROJECT_ID = "adms-320005"
	CONSTANT_DATASET_ID = "australian_search_experience_dataset"
	bq = bigquery.Client()
	keywords = None
	for arg in ["k1", "k2"]:
		if (arg == "k1"):
			keywords = ["Adam Bandt", "Anthony Albanese", "Barnaby Joyce", "Greens", "Labor Party", "Liberal Party", "National Party", "One Nation", "Pauline Hanson","Scott Morrison"]
		if (arg == "k2"):
			keywords = ["Afghanistan", "COP26", "COVID", "Lockdown", "Quarantine", "Renewable energy","Tokyo 2021", "Tokyo Olympics", "Travel rules", "Vaccine", "Feminism"]
		with open(f"browser_and_machine.jsond.temporal.{arg}", "w") as f:
			ii = 0
			for row in simple_cursor(bq, f'''
				SELECT 
				  base.browser_type AS browser_type,
				  base.machine_type AS machine_type,
				  base.user_agent AS user_agent
				FROM `adms-320005.australian_search_experience_dataset.google_news_result` result
				JOIN `adms-320005.australian_search_experience_dataset.google_news_base` base ON base.id = result.base_id
				WHERE keyword in UNNEST({json.dumps(keywords)})
				AND (type = "mobile" or type = "desktop")
				AND list_index <= 9
				  AND time_of_retrieval >= "2021-09-01 00:00:00 Australia/Brisbane"
				  AND time_of_retrieval < "2022-01-01 00:00:00 Australia/Brisbane"'''):
				ii += 1
				if ((ii % 100000) == 0):
					print(ii)
				try:
					f.write(json.dumps({
						"browser_type" : row.get("browser_type"),
						"machine_type" : row.get("machine_type"),
						"user_agent" : row.get("user_agent")
					})+"\n")
				except:
					print(traceback.format_exc())
					ipdb.set_trace()
			f.close()

'''
	This process retrieves the summary of the cumulative weekly users for all operating systems denominated 
	from the overall Australian Search Experience research project

	Note: The weekly_users_by_os.csv file was retrieved from the "ADM+S - The Australian Search Experience" 
	Chrome Web Store Developer Dashboard "Users - Analytics" panel.

'''
excluded_properties = ["Date"]
def process_2():
	# Removing first row for title
	rows = [x for x in open("weekly_users_by_os.csv").read().split("\n")][1:]
	# Isolating second row as properties and remove it
	properties = rows[0].split(","); rows = rows[1:]
	# For each row, construct weekly summary
	weekly_summaries = [{properties[i]:x.split(",")[i] for i in range(len(properties)) 
												if (not properties[i] in excluded_properties)} for x in rows]
	# Determine number of active weeks of operation, and discount those where there was 
	# no activity, as we need to average number of users across all weeks
	n_pre = len(weekly_summaries)
	weekly_summaries = [x for x in weekly_summaries if (sum([int(y) for y in x.values()]) > 0)]
	print(f"Reduced number of weekly summaries from {n_pre} to {len(weekly_summaries)}")

	# Aggregated and divided by number of weeks to produce 'average per week'
	aggregated = {x:(sum([int(y[x]) for y in weekly_summaries])/len(weekly_summaries)) for x in properties if (not x in excluded_properties)}
	print(f"Aggregating across {len(weekly_summaries)} weeks")

	output = {"total_n_weeks_recorded" : n_pre, "total_n_weeks_active" : len(weekly_summaries), "os" : aggregated}
	
	with open("weekly_users_by_os_summarised.json", "w") as f:
		f.write(json.dumps(output, indent=3))
		f.close()

'''
	 Retrieve the breakdown of browser and machine types (that were spoofed) during data acquisition
'''
def process_3():
	rows = list()
	rows.extend(open("browser_and_machine.jsond.temporal.k1").read().split("\n"))
	rows.extend(open("browser_and_machine.jsond.temporal.k2").read().split("\n"))
	rows = [json.loads(x) for x in rows if (len(x) > 0)]

	distribution = dict()
	for x in rows:
		for k in x:
			if (not k in distribution):
				distribution[k] = dict()
			if (not x[k] in distribution[k]):
				distribution[k][x[k]] = int()
			distribution[k][x[k]] += 1

	distribution["totals"] = {x:sum(distribution[x].values()) for x in distribution.keys()}
	for x in distribution:
		if (not x == "totals"):
			distribution[x] = {k:(v/(distribution["totals"][x])) for k,v in distribution[x].items()}


	with open("overall_entries_by_os_browser_and_spoofed_machine.json", "w") as f:
		f.write(json.dumps(distribution, indent=3))
		f.close()

'''
	Reduce user agent strings
'''
def process_4():
	distribution = json.loads(open("overall_entries_by_os_browser_and_spoofed_machine.json").read())
	catchers = {	
		"Windows" : ["Windows"],
		"Mac OS" : ["Mac OS"],
		"Linux" : ["Linux"]
	}
	mappings = dict()
	# For each user agent string
	for this_user_agent_string in distribution["user_agent"]:
		# For each catcher
		found = False
		for k in catchers:
			other_catchers_v = list()
			[other_catchers_v.extend(catchers[k2]) for k2 in catchers if (k2 != k)]
			# If any catcher string corresponds, while none of the alternatives, establish the mapping
			if ((any([x in this_user_agent_string for x in catchers[k]])) 
					and (not any([this_user_agent_string in x for x in other_catchers_v]))):
				mappings[this_user_agent_string] = k
				found = True
				break
		if (not found):
			mappings[this_user_agent_string] = "Other"

	# Now that we have mappings, we can reduce the distribution

	reestablished = dict()
	for k in distribution["user_agent"]:
		if (not mappings[k] in reestablished):
			reestablished[mappings[k]] = float()
		reestablished[mappings[k]] += distribution["user_agent"][k]

	distribution["os"] = reestablished
	distribution["user_agent_mappings"] = mappings

	with open("overall_entries_by_os_browser_and_spoofed_machine.json", "w") as f:
		f.write(json.dumps(distribution, indent=3))
		f.close()


if (__name__ == "__main__"):
	process_4()

















