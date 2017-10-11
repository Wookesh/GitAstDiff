#!/usr/bin/env python

import redis
import numpy as np
import matplotlib.pyplot as plt
import collections

client = redis.StrictRedis()

def main():

	# data = client.hgetall('steamnations:master:aa5310e8')
	# data = client.hgetall('fastText:master:836e5362')
	data = client.hgetall('leveldb:master:54f1fd7e')
	adds = {}
	dels = {}
	for name, v in data.items():
		if name.startswith('avg_add'):
			v = float(v)
			adds[v] = adds.get(v, 0) + 1
		if name.startswith('avg_del'):
			v = float(v)
			dels[v] = dels.get(v, 0) + 1
	sorted_adds = collections.OrderedDict(sorted(adds.items()))
	sorted_dels = collections.OrderedDict(sorted(dels.items()))
	plt.plot(sorted_adds.keys(), sorted_adds.values(), sorted_dels.keys(), sorted_dels.values())

	plt.show()


if __name__ == '__main__':
	main()
