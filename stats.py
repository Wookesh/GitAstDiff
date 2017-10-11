import redis

client = redis.StrictRedis()


def gather_stats(storage, project, first, last, phase, commits_no):
	key = ":".join([project, first[:8], last[:8]])
	count = len(storage.ordered_data)
	client.hset(key, 'count', count)

	avg, min, max = 0.0, commits_no, 0
	for name, history in storage.ordered_data.items():
		rev = history.head
		local_size = 0
		local_avg_lines_added = 0
		local_avg_lines_deleted = 0
		while rev is not None:
			local_size += 1
			next = rev.parents[0] if len (rev.parents) > 0 else None
			if next is not None:
				diff = rev.function.diffLCS(next.function, 0)
				for (mode, _) in diff:
					if mode == 1:
						local_avg_lines_added += 1
					if mode == 2:
						local_avg_lines_deleted += 1
			rev = next
		client.hset(key, 'avg_added:%s' % name, float(local_avg_lines_added) / float(local_size))
		client.hset(key, 'avg_deleted:%s' % name, float(local_avg_lines_deleted) / float(local_size))

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
	client.hset(key, 'phase', phase)
	client.hset(key, 'commits', commits_no)

