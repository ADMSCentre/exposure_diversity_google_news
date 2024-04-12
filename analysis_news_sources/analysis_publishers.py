'''
	This script is responsible for the implementation of the publishers analysis.

	The script is divided into 10 processes that are responsible for retrieving and transforming the Australian Search Experience
	Google News publisher data, as well as visualising the data as bar graphs, treemaps, and pie charts
'''
import re
import os
import sys
import tld
import time
import base64
import ipdb
import json
import math
import boto3
import whois
import scipy
import socket
import botocore
import datetime
import squarify
import itertools
import traceback
import statistics
import matplotlib
import numpy as np
import pandas as pd
from geolite2 import geolite2
from datetime import datetime
import matplotlib.pyplot as plt
from google.cloud import bigquery
from urllib.parse import urlparse
from botocore.client import Config
import matplotlib.colors as colors
from collections import defaultdict
import matplotlib.patches as mpatches

verbose = False

def isolate_article_str(input_str):
	try:
		return re.findall(r'(?<=\.\/articles\/).*?(?=\?)', input_str)[0]
	except:
		if verbose:
			print(traceback.format_exc())
		return None

def remove_non_url_characters(input_str):
	try:
		return re.findall(r'[a-zA-Z0-9\-\.\_\~\:\/\?\#\[\]\@\!\$\&\'\(\)\*\+\,\;\%\=]+',input_str)[0]
	except:
		if verbose:
			print(traceback.format_exc())
		return None


def base64_decode_str(input_str):
	try:
		return base64.b64decode(input_str + "==").decode("utf-8", errors='ignore')
	except:
		if verbose:
			print(traceback.format_exc())
		return None

def isolate_uo_parameter(input_str):
	try:
		return re.findall(r'(?<=uo\=).*?(?=\&|\_)', input_str)[0]
	except:
		if verbose:
			print(traceback.format_exc())
		return None

def isolate_url_raw(input_str):
	try:
		return re.findall(r'(http).*?(?=http)', input_str)[0]
	except:
		if verbose:
			print(traceback.format_exc())
		return None

def isolate_underscore_split(input_str):
	try:
		return input_str.split("_")[0]
	except:
		if verbose:
			print(traceback.format_exc())
		return None

def postprocess_url(this_url):
	if (this_url is not None):
		recognised_prefixes = ["http"]
		for this_prefix in recognised_prefixes:
			if (this_prefix in this_url):
				try:
					return re.findall(f'{this_prefix}.*?$', this_url)[0]
				except:
					pass
	return None

def replace_wildcard_a(this_url):
	return this_url.replace("AUi_", str())

def select_url(urls):
	for url in urls:
		if url is not None:
			return url
	return None

def process_country_code(domain_datastructure, domain_name):
	try:
		if (domain_datastructure[domain_name]["country"] == "Malaysia"):
			return "MY"
		elif (domain_datastructure[domain_name]["country"] == "Sweden"):
			return "SE"
		elif (type(domain_datastructure[domain_name]["country"]) is list):
			return domain_datastructure[domain_name]["country"][0]
		elif (domain_datastructure[domain_name]["country"] in ["REDACTED FOR PRIVACY", "20036"]):
			return None
		else:
			return domain_datastructure[domain_name]["country"]
	except:
		return None

def decode_google_article_str(input_str):
	attempt = select_url([
			postprocess_url(remove_non_url_characters(base64_decode_str(isolate_article_str(input_str)))),
			postprocess_url(remove_non_url_characters(base64_decode_str(isolate_uo_parameter(input_str)))),
			postprocess_url(remove_non_url_characters(isolate_url_raw(base64_decode_str(input_str)))),
			postprocess_url(remove_non_url_characters(base64_decode_str(isolate_underscore_split(isolate_article_str(input_str))))),
			postprocess_url(remove_non_url_characters(base64_decode_str(isolate_uo_parameter(replace_wildcard_a(input_str)))))
		])
	if (attempt is not None):
		return attempt
	else:
		return input_str

'''
	Simple Google BQ cursor function
'''
def simple_cursor(arg_statement):
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
		with open(f"output.jsond.temporal.{arg}", "w") as f:
			ii = 0
			for row in simple_cursor(f'''
				SELECT 
					base.keyword AS keyword, 
					result.list_index AS list_index, 
					result.source_url AS source_url, 
					result.publisher AS publisher, 
					base.hash_key AS activation_code, 
					base.time_of_retrieval AS time_of_retrieval
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
						"keyword" : row.get("keyword"),
						"list_index" : row.get("list_index"),
						"source_url" : row.get("source_url"),
						"publisher" : row.get("publisher"),
						"activation_code" : row.get("activation_code"),
						"time_of_retrieval" : int(row.get("time_of_retrieval").timestamp())
					})+"\n")
				except:
					print(traceback.format_exc())
					ipdb.set_trace()
			f.close()

'''
	The second process assesses the dataset, and extracts all first level domains 
'''
def process_2():
	ii = 0
	subdomains = list()
	for y in ["k1", "k2"]:
		this_str = f"output.jsond.temporal.{y}"
		with open(this_str+".p2", "w") as f:
			for entry in open(this_str).read().split("\n"):
				if (len(entry) > 0):
					entry_as_json = json.loads(entry)
					ii += 1
					if ((ii % 1000) == 0):
						print(ii)
					if ("source_url" in entry_as_json) and ("//" in entry_as_json["source_url"]):
						this_field = entry_as_json["source_url"]
						entry_as_json["source_url_fld"] = tld.get_tld(this_field, as_object=True).fld
						ii += 1
						if ((ii % 1000) == 0):
							print(ii)
					f.write(json.dumps(entry_as_json) + "\n")
			f.close()

'''
	This function extracts a domain from its URL
'''
def extract_domain(this_url):
	try:
		this_url_new = urlparse(this_url).netloc
		if (len(this_url_new) == 0):
			return this_url
		else:
			return this_url_new
	except:
		return "UNKNOWN"

def process_3():
	n_retained = int()
	n_failed_no_fld = int()
	n_failed_articles = int()
	all_flds = list()
	# For both of the keyword sets
	for y in ["k1", "k2"]:
		this_str = f"output.jsond.temporal.{y}.p2"
		with open(this_str.replace(".p2",".p3"), "w") as f:
			# For each of the entries
			for entry in open(this_str).read().split("\n"):
				# If the entry is not an empty line
				if (len(entry) > 0):
					# Synthesize it
					entry_as_json = json.loads(entry)
					# If the source URL does not have the './articles' prefix
					if (not entry_as_json["source_url"].startswith("./articles")):
						try:
							# This is wrapped in a try-catch to avoid FLD absence errors
							all_flds.append(entry_as_json["source_url_fld"])
							entry_as_json["source_url_fld"] = entry_as_json["source_url_fld"]
							f.write(json.dumps(entry_as_json) + "\n")
							n_retained += 1
						except:
							n_failed_no_fld += 1
					else:
						n_failed_articles += 1
			f.close()
	all_flds = list(set(all_flds))
	with open("all_flds.json","w") as f:
		f.write(json.dumps(all_flds,indent=3))
		f.close()
	with open("malformed_fld_stats.json","w") as f:
		f.write(json.dumps({
				"n_retained" : n_retained,
				"n_failed_no_fld" : n_failed_no_fld,
				"n_failed_articles" : n_failed_articles
			},indent=3))
		f.close()

def graph_malformed_fld_stats():
	results = json.loads(open(os.path.join(os.getcwd(),"malformed_fld_stats.json")).read())
	plt.figure(figsize=(5, 4))
	plt.ylabel("No. Of Entries")
	plt.xlabel("Entry Evaluation Outcome")
	plt.title("Outcomes Of Entry Evaluations")
	xlabels = {
	    "n_retained" : "Retained",
	    "n_failed_no_fld" : "No FLD",
	    "n_failed_articles" : "Prefixed by './articles'"
	}
	plt.xticks(fontsize=8)
	plt.bar([xlabels[x] for x in results.keys()], results.values(), width=0.3)
	plt.show()


'''
	For a given domain, this function retrieves its registration data from the MaxMind Geolite2 package
'''
def geolite2_getip(domain_str):
	ip = socket.gethostbyname(domain_str.strip())
	reader = geolite2.reader()	  
	output = reader.get(ip)
	result = output['country']['iso_code']
	return output

'''
	This function is a fallback corrector (of domains with SLDs) for the geolite2_getip function
'''
def geolite2_domain_data(domain_str):
	result = None
	try:
		result = geolite2_getip(domain_str)
	except socket.error as msg:
		if len(domain_str) > 2:
			subdomain = domain_str.split('.', 1)[1]
			try:
				result = geolite2_getip(subdomain)
			except:
				pass
	return result

'''
	The fourth process attempts to determine the country of operation associated with each of the publisher's domain URLs

	Note: This step uses a JavaScript snippet (see country_codes.js) to obtain top-level domains suffixes enumerated at the official Wikipedia page: 
	https://en.wikipedia.org/wiki/Country_code_top-level_domain

	The results inform the contents of the 'wiki_country_suffixes.json' file.

'''
wiki_country_suffixes = [x.replace(".",str()) for x in json.loads(open("wiki_country_suffixes.json").read())]
wiki_country_suffixes_unambiguous = [x.replace(".",str()) for x in json.loads(open("wiki_country_suffixes_unambiguous.json").read())]
removed_country_suffixes = [x for x in wiki_country_suffixes if (not x in wiki_country_suffixes_unambiguous)]
def process_4():
	# Firstly, for all FLDs, a WHOIS data lookup is conducted to determine the countries of operation; the results are stored in the whois_fld_data dictionary
	all_flds = json.loads(open("all_flds.json").read())
	'''
	whois_fld_data = dict()
	# For all domains lacking reporting, get WHOIS data
	n_whois_data_accounted = int()
	n_whois_data_unaccounted = int()
	for this_fld in all_flds:
		try:
			w = whois.whois("https://"+this_fld)
			if (not w["country"] in ["DATA REDACTED", "REDACTED FOR PRIVACY", None]):
				whois_fld_data[this_fld] = dict()
				for key in ('name', 'org', 'address', 'city', 'country'):
					whois_fld_data[this_fld][key] = w[key]
			n_whois_data_accounted += 1
		except:
			pass
			n_whois_data_unaccounted += 1
		print(n_whois_data_accounted, n_whois_data_unaccounted)

	with open("whois_fld_data.json", "w") as f:
		f.write(json.dumps(whois_fld_data,indent=3))
		f.close()
	'''
	whois_fld_data = json.loads(open("whois_fld_data.json").read())
	''''
	geolite2_fld_data = dict()
	# For all domains lacking reporting, get WHOIS data
	n_geolite2_data_accounted = int()
	n_geolite2_data_unaccounted = int()
	for this_fld in all_flds:
		try:
			geolite2_fld_data[this_fld] = geolite2_domain_data(this_fld)
			n_geolite2_data_accounted += 1
		except:
			pass
			n_geolite2_data_unaccounted += 1
	with open("geolite2_fld_data.json", "w") as f:
		f.write(json.dumps(geolite2_fld_data,indent=3))
		f.close()
	'''
	geolite2_fld_data = json.loads(open("geolite2_fld_data.json").read())
	'''
	with open("whois_fld_stats.json","w") as f:
		f.write(json.dumps({
				"resolved" : len(whois_fld_data.keys()),
				"unresolved" : len(all_flds)-len(whois_fld_data.keys())
			},indent=3))
		f.close()
	'''
	# Secondly, for all FLDs, country suffixes are assessed, to determine the country of operation for all domains
	country_suffix_data = dict()
	for x in all_flds:
		if (x.split(".")[-1] in wiki_country_suffixes_unambiguous):
			country_suffix_data[x] = x.split(".")[-1].upper()

	with open("country_suffix_data.json", "w") as f:
		f.write(json.dumps(country_suffix_data,indent=3))
		f.close()
	country_suffix_data = json.loads(open("country_suffix_data.json").read())
	# Thirdly, we determine the country of operation for each news outlet, in reference to the entry being assessed
	n_accounted_by = {k:list() for k in ["australian_subpage", "geolite2", "whois", "country_code_tld", "unaccounted"]}
	australia_subpage_indicators = ["/au/", "/australia-news/"]
	for y in ["k1", "k2"]:
		this_str = f"output.jsond.temporal.{y}.p3"
		with open(this_str.replace(".p3",".p4"), "w") as f:
			for entry in open(this_str).read().split("\n"):
				if (len(entry) > 0):
					entry_as_json = json.loads(entry)
					this_entry_country = None
					# If it has an Australian sub-page indicator, resolve the country as Australia
					if (any([(x in entry_as_json["source_url"]) for x in australia_subpage_indicators])):
						this_entry_country = "AU"
						n_accounted_by["australian_subpage"].append(entry_as_json["source_url_fld"])
					elif (entry_as_json["source_url_fld"] in country_suffix_data):
						this_entry_country = country_suffix_data[entry_as_json["source_url_fld"]]
						n_accounted_by["country_code_tld"].append(entry_as_json["source_url_fld"])
					elif ((entry_as_json["source_url_fld"] in geolite2_fld_data) 
							and (geolite2_fld_data[entry_as_json["source_url_fld"]] is not None)
							and ("country" in geolite2_fld_data[entry_as_json["source_url_fld"]]) 
							and ("iso_code" in geolite2_fld_data[entry_as_json["source_url_fld"]]["country"])
							and (geolite2_fld_data[entry_as_json["source_url_fld"]]["country"]["iso_code"] is not None)):
						# Fallback on GeoLite2
						this_entry_country = geolite2_fld_data[entry_as_json["source_url_fld"]]["country"]["iso_code"]
						n_accounted_by["geolite2"].append(entry_as_json["source_url_fld"])
					elif ((entry_as_json["source_url_fld"] in whois_fld_data) 
							and ("country" in whois_fld_data[entry_as_json["source_url_fld"]])
							and (whois_fld_data[entry_as_json["source_url_fld"]]["country"] is not None)):
						# Fallback on WHOIS
						this_entry_country = whois_fld_data[entry_as_json["source_url_fld"]]["country"]
						n_accounted_by["whois"].append(entry_as_json["source_url_fld"])
					else:
						n_accounted_by["unaccounted"].append(entry_as_json["source_url_fld"])
					entry_as_json["country"] = this_entry_country
					f.write(json.dumps(entry_as_json) + "\n")
			f.close()

	with open("country_of_operation_n_accounted_by.json", "w") as f:
		f.write(json.dumps({k:len(list(set(v))) for k,v in n_accounted_by.items()},indent=3))
		f.close()

'''
	This function graphs the distribution of each of the possible methods applied in the analysis for determining countries of operation,
	frequentized by number of times applied.
'''
def graph_country_of_operation_n_accounted_by():
	results = json.loads(open(os.path.join(os.getcwd(),"country_of_operation_n_accounted_by.json")).read())
	plt.figure(figsize=(7, 4))
	plt.ylabel("No. Of Distinct FLDs")
	plt.xlabel("Method Of Evaluation")
	plt.title("Outcomes Of Evaluations For Country Of Operation")
	xlabels = {
			"australian_subpage": "Australian Subpage",
			"geolite2": "GeoLite2",
			"whois": "WHOIS",
			"country_code_tld": "Country-Code TLD",
			"unaccounted": "Unaccounted"
		}
	plt.xticks(fontsize=8)
	plt.bar([xlabels[x] for x in results.keys()], results.values(), width=0.3)
	plt.show()

'''
	This function groups entries by keyword categories, keywords, FLDs, and the condition of whether or not the entries are represented by distinct individuals who submitted
	the data donations, or not.
	
	Note: While this step generates some records that do appear as duplicates, this is a false alarm, and can be crosschecked against the ASE dataset via the following
	SQL command:

	SELECT
	  base.keyword AS keyword, 
	  result.list_index AS list_index, 
	  result.source_url AS source_url, 
	  result.publisher AS publisher, 
	  base.hash_key AS activation_code, 
	  base.time_of_retrieval AS time_of_retrieval, COUNT(*)
	FROM `adms-320005.australian_search_experience_dataset.google_news_result` result
	JOIN `adms-320005.australian_search_experience_dataset.google_news_base` base ON base.id = result.base_id
	WHERE (type = "mobile" or type = "desktop")
	AND list_index <= 9
	AND time_of_retrieval >= "2021-09-01 00:00:00 Australia/Brisbane"
	AND time_of_retrieval < "2022-01-01 00:00:00 Australia/Brisbane"
	GROUP BY
	    keyword, list_index, source_url, publisher, activation_code, time_of_retrieval
	HAVING 
	    COUNT(*) > 1
'''
def process_5():
	entries_by_domain = dict()
	# For both the 'distinct' and 'non-distinct' case (in relation to individuals who submitted the data donations)
	for distinct_activation_codes in [True, False]:
		d = "distinct" if distinct_activation_codes else "nondistinct"
		entries_by_domain[d] = dict()
		# For both keyword sets
		for y in ["k1", "k2"]:
			entries_by_domain[d][y] = dict()
			# Load in the relative file from the previous process
			for entry in open(f"output.jsond.temporal.{y}.p4").read().split("\n"):
				if (len(entry) > 0):
					entry_as_json = json.loads(entry)
					this_keyword = entry_as_json["keyword"]
					if (not this_keyword in entries_by_domain[d][y]):
						entries_by_domain[d][y][this_keyword] = dict()
					# For the given entry, create a unique identifier for the domain, as a composite of its FLD and country of operation
					this_fld_cc_appendage = entry_as_json["source_url_fld"] + f" ({'Unknown' if (not 'country' in entry_as_json) else entry_as_json['country']})"
					if (distinct_activation_codes):
						# In the distinct case, group list indices and times of retrieval of search results for entries by keyword, FLD, AND the unique identifier of 
						# the distinct individuals who submitted the data donations
						if (not this_fld_cc_appendage in entries_by_domain[d][y][this_keyword]):
							entries_by_domain[d][y][this_keyword][this_fld_cc_appendage] = dict()
						if (not entry_as_json["activation_code"] in entries_by_domain[d][y][this_keyword][this_fld_cc_appendage]):
							entries_by_domain[d][y][this_keyword][this_fld_cc_appendage][entry_as_json["activation_code"]] = list()
						entries_by_domain[d][y][this_keyword][this_fld_cc_appendage][entry_as_json["activation_code"]].append({
							"list_index" : entry_as_json["list_index"], 
							"time_of_retrieval" : entry_as_json["time_of_retrieval"]})
					else:
						# In the non-distinct case, group list indices and times of retrieval of search results for entries by keyword, and FLDs
						if (not this_fld_cc_appendage in entries_by_domain[d][y][this_keyword]):
							entries_by_domain[d][y][this_keyword][this_fld_cc_appendage] = list()
						entries_by_domain[d][y][this_keyword][this_fld_cc_appendage].append({
								"activation_code" : entry_as_json["activation_code"],
								"list_index" : entry_as_json["list_index"],
								"time_of_retrieval" : entry_as_json["time_of_retrieval"]
							})
			# In the distinct case, average all list indices donated for each individual
			if (distinct_activation_codes):
				for this_keyword in entries_by_domain[d][y]:
					for this_fld_cc_appendage in entries_by_domain[d][y][this_keyword]:
						entries_by_domain[d][y][this_keyword][this_fld_cc_appendage] = [{
							"activation_code" : k, 
							"list_index" : np.mean([x["list_index"] for x in v]), 
							"time_of_retrieval" : [x["time_of_retrieval"] for x in v]
						} for k,v in entries_by_domain[d][y][this_keyword][this_fld_cc_appendage].items()]
	with open("entries_by_domain.json", "w") as f:
		f.write(json.dumps(entries_by_domain,indent=3))
		f.close()

'''
	This function generates a copy of the "entries_by_domain.json" file that accounts for the temporality of the entries.
'''
def process_6():
	# UNIX timestamps for the bounds of the period of data collection
	window_start = 1630418400 # (i.e., "Wednesday, September 1, 2021 12:00:00 AM GMT+10:00")
	window_end = 1640959200 # (i.e., "Saturday, January 1, 2022 12:00:00 AM GMT+10:00")
	# We decide on the interval of 3 days
	window_interval = 1*24*60*60 # (3 days * 24 hours * 60 minutes * 60 seconds)
	temporal_time_windows = list(range(window_start, window_end+window_interval, window_interval))
	# Load in the 'entries_by_domain' file
	entries_by_domain = json.loads(open("entries_by_domain.json").read())
	entries_by_domain_temporal = dict()
	# For both the distinct and nondistinct cases
	for d in ["distinct", "nondistinct"]:
		entries_by_domain_temporal[d] = dict()
		# For each of the keyword categories
		for y in ["k1", "k2"]:
			entries_by_domain_temporal[d][y] = dict()
			# For each of the keywords
			for this_keyword in entries_by_domain[d][y]:
				entries_by_domain_temporal[d][y][this_keyword] = dict()
				# For each of the FLDs
				for fld_cc in entries_by_domain[d][y][this_keyword]:
					entries_by_domain_temporal[d][y][this_keyword][fld_cc] = dict()
					# For each of the intervals
					for interval in temporal_time_windows:
						entries_by_domain_temporal[d][y][this_keyword][fld_cc][interval] = list()
						# For each entry
						for entry in entries_by_domain[d][y][this_keyword][fld_cc]:
							# If the entry is a list (such as in the distinct case), and one such time of observation for the entry was observed in the given time interval
							#
							# OR
							#
							# If the entry is an integer (such as in the nondistinct case), and the time of observation for the entry was observed in the given time interval
							if (((type(entry["time_of_retrieval"]) is list) and (any([(x >= interval and x < interval+window_interval) for x in entry["time_of_retrieval"]]))) 
								or ((type(entry["time_of_retrieval"]) is int) and (entry["time_of_retrieval"] >= interval and entry["time_of_retrieval"] < interval+window_interval))):
								# Then include a copy of this entry within the interval
								time_of_retrieval = (entry["time_of_retrieval"] if (type(entry["time_of_retrieval"]) is int) 
														else [x for x in entry["time_of_retrieval"] if (x >= interval and x < interval+window_interval)])
								entries_by_domain_temporal[d][y][this_keyword][fld_cc][interval].append({
										"list_index" : entry["list_index"],
										"activation_code" : entry["activation_code"],
										"time_of_retrieval" : time_of_retrieval
									})
	with open("entries_by_domain_temporal.json", "w") as f:
		f.write(json.dumps(entries_by_domain_temporal,indent=3))
		f.close()

def normality_test(list_indices):
	NORMALITY_TEST_ALPHA = 0.05
	MINIMUM_ELEMENTS_IN_SAMPLE = 20
	try:
		if (len(list_indices) < MINIMUM_ELEMENTS_IN_SAMPLE):
			raise Exception() # kurtosis test constraint
		normality_test = scipy.stats.normaltest(list_indices)
		normality_test_outcome = {
				"statistic" : normality_test.statistic,
				"p_value" : normality_test.pvalue,
				"success" : bool((normality_test.pvalue < NORMALITY_TEST_ALPHA))
			}
	except:
		normality_test_outcome = {
				"success" : False
			}
	return normality_test_outcome

'''
	This function determines the confidence of the results, by implementing normality test
'''
def process_7():
	# Non-temporal case
	entries_by_domain = json.loads(open("entries_by_domain.json").read())
	entries_by_domain_summarised = dict()
	# For distinct/non-distinct results
	for d in entries_by_domain:
		entries_by_domain_summarised[d] = dict()
		# For each of the keyword categories
		for y in ["k1", "k2"]:
			entries_by_domain_summarised[d][y] = dict()
			# For each keyword
			for this_keyword in entries_by_domain[d][y]:
				entries_by_domain_summarised[d][y][this_keyword] = dict()
				# For each FLD
				for fld_cc in entries_by_domain[d][y][this_keyword]:
					# Retrieve the list indices
					list_indices = [x["list_index"] for x in entries_by_domain[d][y][this_keyword][fld_cc]]
					# Calculate the average, frequency, and normality test outcome of the results
					entries_by_domain_summarised[d][y][this_keyword][fld_cc] = {
							"avg" : np.mean(list_indices),
							"frq" : len(list_indices),
							"normality_test_outcome" : normality_test(list_indices)
						}
	# Temporal case
	entries_by_domain_temporal = json.loads(open("entries_by_domain_temporal.json").read())
	entries_by_domain_temporal_summarised = dict()
	# For distinct/non-distinct results
	for d in entries_by_domain_temporal:
		entries_by_domain_temporal_summarised[d] = dict()
		# For each of the keyword categories
		for y in ["k1", "k2"]:
			entries_by_domain_temporal_summarised[d][y] = dict()
			# For each keyword
			for this_keyword in entries_by_domain[d][y]:
				entries_by_domain_temporal_summarised[d][y][this_keyword] = dict()
				# For each FLD
				for fld_cc in entries_by_domain_temporal[d][y][this_keyword]:
					entries_by_domain_temporal_summarised[d][y][this_keyword][fld_cc] = dict()
					# For each interval
					for interval in entries_by_domain_temporal[d][y][this_keyword][fld_cc]:
						# Retrieve the list indices
						list_indices = [x["list_index"] for x in entries_by_domain_temporal[d][y][this_keyword][fld_cc][interval]]
						# Calculate the average, frequency, and normality test outcome of the results
						entries_by_domain_temporal_summarised[d][y][this_keyword][fld_cc][interval] = {
								"avg" : np.mean(list_indices),
								"frq" : len(list_indices),
								"normality_test_outcome" : normality_test(list_indices)
							}
	with open("entries_by_domain_summarised.json", "w") as f:
		f.write(json.dumps(entries_by_domain_summarised,indent=3))
		f.close()
	with open("entries_by_domain_temporal_summarised.json", "w") as f:
		f.write(json.dumps(entries_by_domain_temporal_summarised,indent=3))
		f.close()
			

# TODO - invetsigate empty time windows re process_6

# TODO 
def process_8():
	entries_by_domain_summarised = json.loads(open("entries_by_domain_summarised.json").read())
	entries_by_domain_temporal_summarised = json.loads(open("entries_by_domain_temporal_summarised.json").read())
	for excluding_normality_test_failures in [True,False]:
		for d in ["distinct", "nondistinct"]:
			for y in ["k1", "k2"]:
				# NON TEMPORAL agnostic CASE
				master_graphable_dataset  = dict()
				master_graphable_dataset_adjusted = dict()
				for tp in [str(), "_temporal"]:
					master_graphable_dataset[tp] = dict()
					master_graphable_dataset_adjusted[tp] = dict()
					for ka in ["graphable_keyword_agnostic", "graphable_keyword_relative"]:
						master_graphable_dataset[tp][ka] = dict()
						master_graphable_dataset_adjusted[tp][ka] = dict()
						# 'Non-temporal' 'graphable_keyword_agnostic' case
						if (tp == str()):
							if (ka == "graphable_keyword_agnostic"):
								for this_stat in ["frq", "avg"]:
									master_graphable_dataset[tp][ka][this_stat] = dict()
									for this_keyword in entries_by_domain_summarised[d][y]:
										for this_fld in entries_by_domain_summarised[d][y][this_keyword]:
											this_entry = entries_by_domain_summarised[d][y][this_keyword][this_fld]
											if ((not excluding_normality_test_failures) or (this_entry["normality_test_outcome"]["success"])):
												if (not np.isnan(this_entry[this_stat])):
													if (not this_fld in master_graphable_dataset[tp][ka][this_stat]):
														master_graphable_dataset[tp][ka][this_stat][this_fld] = list()
													master_graphable_dataset[tp][ka][this_stat][this_fld].append(this_entry[this_stat])
								for this_stat in ["frq", "avg"]:
									master_graphable_dataset_adjusted[tp][ka][this_stat] = dict()
									for this_fld in master_graphable_dataset[tp][ka][this_stat]:
										v = sum(master_graphable_dataset[tp][ka][this_stat][this_fld])
										if (v > 0):
											if (this_stat == "frq"):
												master_graphable_dataset_adjusted[tp][ka][this_stat][this_fld] = v
											else:
												master_graphable_dataset_adjusted[tp][ka][this_stat][this_fld] = np.mean(master_graphable_dataset[tp][ka][this_stat][this_fld])
									master_graphable_dataset_adjusted[tp][ka][this_stat] = {k: v for k, v in sorted(master_graphable_dataset_adjusted[tp][ka][this_stat].items(), 
																															key=lambda item: item[1], reverse=(this_stat == "frq"))}
									master_graphable_dataset_adjusted[tp][ka][this_stat] = {
											"domain" : list(master_graphable_dataset_adjusted[tp][ka][this_stat].keys()), 
											this_stat : list(master_graphable_dataset_adjusted[tp][ka][this_stat].values())
										}
							# 'Non-temporal' 'graphable_keyword_relative' case
							if (ka == "graphable_keyword_relative"):
								for this_keyword in entries_by_domain_summarised[d][y]:
									master_graphable_dataset[tp][ka][this_keyword] = dict()
									for this_stat in ["frq", "avg"]:
										master_graphable_dataset[tp][ka][this_keyword][this_stat] = dict()
										for this_fld in entries_by_domain_summarised[d][y][this_keyword]:
											this_entry = entries_by_domain_summarised[d][y][this_keyword][this_fld]
											if ((not excluding_normality_test_failures) or (this_entry["normality_test_outcome"]["success"])):
												if (not np.isnan(this_entry[this_stat])):
													if (not this_fld in master_graphable_dataset[tp][ka][this_keyword][this_stat]):
														master_graphable_dataset[tp][ka][this_keyword][this_stat][this_fld] = list()
													master_graphable_dataset[tp][ka][this_keyword][this_stat][this_fld].append(this_entry[this_stat])
								for this_keyword in entries_by_domain_summarised[d][y]:
									master_graphable_dataset_adjusted[tp][ka][this_keyword] = dict()
									for this_stat in ["frq", "avg"]:
										master_graphable_dataset_adjusted[tp][ka][this_keyword][this_stat] = dict()
										for this_fld in master_graphable_dataset[tp][ka][this_keyword][this_stat]:
											v = sum(master_graphable_dataset[tp][ka][this_keyword][this_stat][this_fld])
											if (v > 0):
												if (this_stat == "frq"):
													master_graphable_dataset_adjusted[tp][ka][this_keyword][this_stat][this_fld] = v
												else:
													master_graphable_dataset_adjusted[tp][ka][this_keyword][this_stat][this_fld] = np.mean(master_graphable_dataset[tp][ka][this_keyword][this_stat][this_fld])
										master_graphable_dataset_adjusted[tp][ka][this_keyword][this_stat] = {k: v for k, v in sorted(master_graphable_dataset_adjusted[tp][ka][this_keyword][this_stat].items(), 
																																							key=lambda item: item[1], reverse=(this_stat == "frq"))}
										master_graphable_dataset_adjusted[tp][ka][this_keyword][this_stat] = {
											"domain" : list(master_graphable_dataset_adjusted[tp][ka][this_keyword][this_stat].keys()), 
											this_stat : list(master_graphable_dataset_adjusted[tp][ka][this_keyword][this_stat].values())
										}
						elif (tp == "_temporal"):
							if (ka == "graphable_keyword_agnostic"):
								for this_stat in ["frq", "avg"]:
									master_graphable_dataset[tp][ka][this_stat] = dict()
									for this_keyword in entries_by_domain_temporal_summarised[d][y]:
										for this_fld in entries_by_domain_temporal_summarised[d][y][this_keyword]:
											for interval in entries_by_domain_temporal_summarised[d][y][this_keyword][this_fld]:
												if (not interval in master_graphable_dataset[tp][ka][this_stat]):
													master_graphable_dataset[tp][ka][this_stat][interval] = dict()
												this_entry = entries_by_domain_temporal_summarised[d][y][this_keyword][this_fld][interval]
												if ((not excluding_normality_test_failures) or (this_entry["normality_test_outcome"]["success"])):
													if (not np.isnan(this_entry[this_stat])):
														if (not this_fld in master_graphable_dataset[tp][ka][this_stat][interval]):
															master_graphable_dataset[tp][ka][this_stat][interval][this_fld] = list()
														master_graphable_dataset[tp][ka][this_stat][interval][this_fld].append(this_entry[this_stat])
								for this_stat in ["frq", "avg"]:
									master_graphable_dataset_adjusted[tp][ka][this_stat] = dict()
									for interval in master_graphable_dataset[tp][ka][this_stat]:
										master_graphable_dataset_adjusted[tp][ka][this_stat][interval] = dict()
										for this_fld in master_graphable_dataset[tp][ka][this_stat][interval]:
											v = sum(master_graphable_dataset[tp][ka][this_stat][interval][this_fld])
											if (v > 0):
												if (this_stat == "frq"):
													master_graphable_dataset_adjusted[tp][ka][this_stat][interval][this_fld] = v
												else:
													master_graphable_dataset_adjusted[tp][ka][this_stat][interval][this_fld] = np.mean(master_graphable_dataset[tp][ka][this_stat][interval][this_fld])
										master_graphable_dataset_adjusted[tp][ka][this_stat][interval] = {k: v for k, v in sorted(master_graphable_dataset_adjusted[tp][ka][this_stat][interval].items(), 
																																key=lambda item: item[1], reverse=(this_stat == "frq"))}
										master_graphable_dataset_adjusted[tp][ka][this_stat][interval] = {
												"domain" : list(master_graphable_dataset_adjusted[tp][ka][this_stat][interval].keys()), 
												this_stat : list(master_graphable_dataset_adjusted[tp][ka][this_stat][interval].values())
											}
							if (ka == "graphable_keyword_relative"):
								for this_keyword in entries_by_domain_summarised[d][y]:
									master_graphable_dataset[tp][ka][this_keyword] = dict()
									for this_stat in ["frq", "avg"]:
										if (not this_stat in master_graphable_dataset[tp][ka][this_keyword]):
											master_graphable_dataset[tp][ka][this_keyword][this_stat] = dict()
										for this_fld in entries_by_domain_temporal_summarised[d][y][this_keyword]:
											for interval in entries_by_domain_temporal_summarised[d][y][this_keyword][this_fld]:
												if (not interval in master_graphable_dataset[tp][ka][this_keyword][this_stat]):
													master_graphable_dataset[tp][ka][this_keyword][this_stat][interval] = dict()
												this_entry = entries_by_domain_temporal_summarised[d][y][this_keyword][this_fld][interval]
												if ((not excluding_normality_test_failures) or (this_entry["normality_test_outcome"]["success"])):
													if (not np.isnan(this_entry[this_stat])):
														if (not this_fld in master_graphable_dataset[tp][ka][this_keyword][this_stat][interval]):
															master_graphable_dataset[tp][ka][this_keyword][this_stat][interval][this_fld] = list()
														master_graphable_dataset[tp][ka][this_keyword][this_stat][interval][this_fld].append(this_entry[this_stat])
								for this_keyword in entries_by_domain_summarised[d][y]:
									master_graphable_dataset_adjusted[tp][ka][this_keyword] = dict()
									for this_stat in ["frq", "avg"]:
										master_graphable_dataset_adjusted[tp][ka][this_keyword][this_stat] = dict()
										for interval in master_graphable_dataset[tp][ka][this_keyword][this_stat]:
											master_graphable_dataset_adjusted[tp][ka][this_keyword][this_stat][interval] = dict()
											for this_fld in master_graphable_dataset[tp][ka][this_keyword][this_stat][interval]:
												v = sum(master_graphable_dataset[tp][ka][this_keyword][this_stat][interval][this_fld])
												if (v > 0):
													if (this_stat == "frq"):
														master_graphable_dataset_adjusted[tp][ka][this_keyword][this_stat][interval][this_fld] = v
													else:
														master_graphable_dataset_adjusted[tp][ka][this_keyword][this_stat][interval][this_fld] = np.mean(master_graphable_dataset[tp][ka][this_keyword][this_stat][interval][this_fld])
											master_graphable_dataset_adjusted[tp][ka][this_keyword][this_stat][interval] = {k: v for k, v in sorted(master_graphable_dataset_adjusted[tp][ka][this_keyword][this_stat][interval].items(), 
																																	key=lambda item: item[1], reverse=(this_stat == "frq"))}
											master_graphable_dataset_adjusted[tp][ka][this_keyword][this_stat][interval] = {
													"domain" : list(master_graphable_dataset_adjusted[tp][ka][this_keyword][this_stat][interval].keys()), 
													this_stat : list(master_graphable_dataset_adjusted[tp][ka][this_keyword][this_stat][interval].values())
												}
				n = ".normality" if (excluding_normality_test_failures) else ""
				with open(f"master_graphable_dataset.{y}.{d}{n}.json", "w") as f:
					f.write(json.dumps(master_graphable_dataset_adjusted,indent=3))
					f.close()

def to_jsonlists():
	list_of_entries = list()
	for n in [".normality",str()]:
		for d in ["distinct", "nondistinct"]:
			for y in ["k1", "k2"]:
				this_dataset = json.loads(open(f"master_graphable_dataset.{y}.{d}{n}.json").read())
				for t in [str()]:
					for ag in ["graphable_keyword_agnostic", "graphable_keyword_relative"]:
						if (ag == "graphable_keyword_agnostic"):
							for stat in ["frq", "avg"]:
								for domain in this_dataset[t][ag][stat]["domain"]:
									list_of_entries.append({
											"normality_type" : n,
											"distinct_type" : d,
											"keyword_category" : y,
											"temporal" : False,
											"keyword_agnostic" : (ag == "graphable_keyword_agnostic"),
											"keyword" : None,
											"stat" : stat,
											"domain" : domain,
											"value" : this_dataset[t][ag][stat][stat][this_dataset[t][ag][stat]["domain"].index(domain)]
										})
						else:
							for keyword in this_dataset[t][ag]:
								for stat in ["frq", "avg"]:
									for domain in this_dataset[t][ag][keyword][stat]["domain"]:
										list_of_entries.append({
												"normality_type" : n,
												"distinct_type" : d,
												"keyword_category" : y,
												"temporal" : False,
												"keyword_agnostic" : (ag == "graphable_keyword_agnostic"),
												"keyword" : keyword,
												"stat" : stat,
												"domain" : domain,
												"value" : this_dataset[t][ag][keyword][stat][stat][this_dataset[t][ag][keyword][stat]["domain"].index(domain)]
											})
	with open(f"list_of_entries.jsond", "w") as f:
		for x in list_of_entries:
			f.write(json.dumps(x)+"\n")
		f.close()



if (__name__ == "__main__"):
	#process_4()
	#process_5()
	process_6()
	process_7()
	process_8()
	to_jsonlists()

geolite2.close()


# apprehensive of CC method