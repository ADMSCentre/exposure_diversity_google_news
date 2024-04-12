import ipdb
rows = open("bquxjob_66350a56_18ec3217d94.csv").read().split("\n")[1:]

rows = [[x[:x.index("(")-1],float(x.split(",")[1])] for x in rows]

distribution = dict()
for x in rows:
	if (not x[0] in distribution):
		distribution[x[0]] = float()
	distribution[x[0]] += x[1]


distribution = {k: v for k, v in sorted(distribution.items(), reverse=True, key=lambda item: item[1])}

distribution_values = list(distribution.values())

top_n = 300

top_share = sum(distribution_values[:top_n])
bottom_share = sum(distribution_values[top_n:])

top_share_pct = top_share/(top_share+bottom_share)


ipdb.set_trace()

#for x in rows:
