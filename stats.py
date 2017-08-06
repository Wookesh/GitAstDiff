import redis

client = redis.StrictRedis()


def gather_stats(storage, project, first, last):
	key = ":".join([project, first[:8], last[:8]])
	count = len(storage.ordered_data)
	client.hset(key, 'count', count)

	avg, min, max = 0.0, 100, 0
	for name, history in storage.ordered_data.items():
		rev = history.head
		local_size = 0
		while rev is not None:
			local_size += 1
			rev = rev.parents[0] if len (rev.parents) > 0 else None
		print name, local_size
		if local_size > max:
			max = local_size
		if local_size < min:
			min = local_size
		avg += float(local_size)
	if count > 0:
		avg = avg / float(count)
	else:
		avg = 0

	client.hset(key, 'avg', avg)
	client.hset(key, 'max', max)
	client.hset(key, 'min', min)

