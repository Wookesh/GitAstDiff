
import clang.cindex


class Color(object):
	New	 = 0
	Changed = 1
	Same	= 2
	Removed = 3
	Differ  = 4
	Moved   = 5

def hashNode(node):
	tokens = [t.spelling for t in node.get_tokens()]
	tokens.append(node.kind)

	return hash(frozenset(tokens))


def children(node):
	return [c for c in node.get_children()]


def createHashMap(node, hashMap):

	h = hashNode(node)

	if h not in hashMap:
		hashMap[h] = list()
	hashMap[h].append(node)

	for c in node.get_children():
		createHashMap(c, hashMap)


def compareTokens(a, b):
	if a.kind == b.kind:
		a_tokens = list((t.spelling for t in a.get_tokens()))
		b_tokens = list((t.spelling for t in b.get_tokens()))

		a_tokens_set = set(a_tokens)
		b_tokens_set = set(b_tokens)

		removed_tokens = [e for e in b_tokens if e not in a_tokens_set]
		new_tokens = [e for e in a_tokens if e not in b_tokens_set]

		if len(removed_tokens) == 0 and len(new_tokens) == 0:
			return True
	return False


# find what's new
def findDiff(old, new):

	i, j = 0, 0
	old_children = [e for e in old.get_children()]
	new_children = [e for e in new.get_children()]
	while i < len(old_children) and j < len(new_children):
		o = old_children[i]
		n = new_children[j]
		o.color = None
		n.color = None

		if not compareTokens(o, n):
			print "different tokens"
			# find new nodes
			for k in xrange(j, len(new_children)):
				if len(findInTree(o, new_children[k])) > 0:
					o.color = Color.Moved
					i += 1
					j += 1
					break
			else:
				o.color = Color.Removed
				i += 1

			# check new tree
			# for k in xrange(i, len(old_children)):
			# 	if len(findInTree(n, old_children[k])) > 0:
			# 		n.color = Color.Moved
			# 		break
			# else:
			# 	n.color = Color.New
		else:
			o.color = Color.Same
			n.color = Color.Same
		print o.color, o.kind, n.color, n.kind

def MarkNodes(old, new):
	return markNodes(old, new, dict())

def markNodes(old, new, mapped):
	old_children = [e for e in old.get_children()]
	new_children = [e for e in new.get_children()]
	if old.kind == new.kind:
		# i = 0
		# j = 0
		# while i < len(old_children) and j < len(new_children):
		# 	o = old_children[i]
		# 	n = new_children[j]
			# n.color = None

		for o in old_children:
			for n in new_children:
				if getattr(n, 'color', None) != None:
					continue
				if o.kind == n.kind:
					if not compareTokens(o, n):
						pass
					else:
						n.color = Color.Same
						break
					# check if match

	return


class Node(object):

	def __init__(self, node, r_depth=0):
		self.node = node
		self.children = list()
		self.label = list()
		self.r_depth = r_depth
		for c in node.get_children():
			child = Node(c, r_depth + 1)
			self.children.append(child)
			self.label.append(len(child.label))

	def printLabels(self):
		print self.label
		for c in self.children:
			c.printLabels()

	def compByLabels(self, other):
		if len(self.label) != len(other.label):
			return False
		for i in xrange(0, len(self.label)):
			if not self.children[i].compByLabels(other.children[i]):
				return False
		return True



class Grouper(object):

	def __init__(self, node):
		self.__clear()
		self.__group(node, list())
		for k, v in self.vars.iteritems():
			self.vars[k] = sorted(v, cmp=compareLists)

	def __clear(self):
		self.data = dict()
		self.vars = dict()
		self.global_used = dict()

	def __group(self, node, path):
		path_copy = list(path)
		path_copy.append(node.kind)
		if self.data.get(node.kind, None) == None:
			self.data[node.kind] = list()
		self.data[node.kind].append(node)

		print node.kind

		if node.kind == clang.cindex.CursorKind.DECL_REF_EXPR:
			if self.vars.get(node.displayname, None) == None:
				print "\tUNDEFINED:" + node.displayname
			else:
				print "\tDEFINED:" + node.displayname
				self.vars[node.displayname].append(path_copy)
		if node.kind == clang.cindex.CursorKind.VAR_DECL:
			self.vars[node.displayname] = list()
			print "\t" + node.displayname

		for c in node.get_children():
			self.__group(c, path_copy)

	def matchVars(self, other):
		matched = dict()
		for var, positions in other.vars.iteritems():
			if var not in self.vars:
				possible = None
				possible_value = 0.0
				for new, new_positions in self.vars.iteritems():
					# if new not in matched:

					similarity_val = 0.0
					for i in xrange(0, len(positions)):
						similarity_val += similarity(positions[i], new_positions[i])
					similarity_val = similarity_val / len(positions)

					if similarity_val > 60 and similarity_val > possible_value:
						possible = new
						possible_value = similarity_val
				if possible != None:
					matched[possible] = var
					print "matched", var, possible
				else:
					print "cannot match", var
			else:
				print "matched direct", var, var 
				matched[var] = var


def similarity(pathA, pathB):
	z = zip(pathA, pathB)
	s = 0.0
	for a, b in z:
		if a == b:
			s += 1
	return s * 100/len(z)


def compareLists(a, b):
	for i in xrange(0, min(len(a), len(b))):
		if a[i] < b[i]:
			return True
		elif a[i] > b[i]:
			return False
		i += 1
	return True

def findInTree(node, tree):

	possible = list()

	def iter(node, tree):
		if compareTokens(node, tree):
			possible.append(tree)

		else:
			for child in tree.get_children():
				iter(node, child)


	iter(node, tree)

	return possible
