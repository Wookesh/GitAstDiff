#!/usr/bin/env python
import git
import clang.cindex as ci
import argparse
from ctypes.util import find_library
import os
import ui
import logging
import types
import Queue
import collections
import stats

def init():
	logging.basicConfig(filename="tool.log", level=logging.DEBUG)
	ci.Config.set_library_file(find_library('clang-3.8'))

class Color(object):
	Same	= 0
	Removed = 1
	New	    = 2
	Marked  = 4
	# Differ  = 4

class Mode():
	Both = 0
	New  = 1
	Old  = 2

ModeToColor = {
	Mode.Both: Color.Marked,
	Mode.New:  Color.New,
	Mode.Old:  Color.Removed,
}


scoreComparitionCache = dict()
def cacheResult(func):
	def wrapper(*args, **kwargs):
		node = kwargs['node']
		o_node = kwargs['o_node']
		result = scoreComparitionCache.get(node.hash + '' + o_node.hash, None)
		if result is None:
			result = func(args, kwargs)
			scoreComparitionCache[node.hash + '' + o_node.hash] = result
			return result
		return result
	return wrapper


def LCS(X, Y):
	m = len(X)
	n = len(Y)
	# An (m+1) times (n+1) matrix
	C = [[0] * (n + 1) for _ in range(m + 1)]
	for i in range(1, m+1):
		for j in range(1, n+1):
			if X[i-1] == Y[j-1]: 
				C[i][j] = C[i-1][j-1] + 1
			else:
				C[i][j] = max(C[i][j-1], C[i-1][j])
	return C


def printDiff(C, X, Y, i, j):
	if i > 0 and j > 0 and X[i-1] == Y[j-1]:
		printDiff(C, X, Y, i-1, j-1)
		print "  " + X[i-1]
	else:
		if j > 0 and (i == 0 or C[i][j-1] >= C[i-1][j]):
			printDiff(C, X, Y, i, j-1)
			print "+ " + Y[j-1]
		elif i > 0 and (j == 0 or C[i][j-1] < C[i-1][j]):
			printDiff(C, X, Y, i-1, j)
			print "- " + X[i-1]


def getDiff(C, X, Y, i, j, diff, mode=Mode.Both):
	if i > 0 and j > 0 and X[i-1] == Y[j-1]:
		getDiff(C, X, Y, i-1, j-1, diff, mode)
		diff.append((0, X[i-1]))
	else:
		if j > 0 and (i == 0 or C[i][j-1] >= C[i-1][j]):
			getDiff(C, X, Y, i, j-1, diff, mode)
			if mode in [Mode.Both, Mode.New]:
				diff.append((2, Y[j-1]))
		elif i > 0 and (j == 0 or C[i][j-1] < C[i-1][j]):
			getDiff(C, X, Y, i-1, j, diff, mode)
			if mode in [Mode.Both, Mode.Old]:
				diff.append((1, X[i-1]))


def desc(node, tab=""):
	print tab, node.kind
	for i in node.children:
		desc(i, tab+"\t")


class Object(object):
	def __init__(self, node, name, sourceFile):
		self.node = node
		self.name = name
		self._start_offset = node.extent.start.offset
		self.prepare(self.node, sourceFile)


	def prepare(self, node, sourceText):
		global mode
		if mode == 'struct':
			node.text = sourceText[node.extent.start.offset:node.extent.end.offset]
		if getattr(node, 'text', None) is None:
			node.text = sourceText[node.extent.start.offset:node.extent.end.offset]
			# from pprint import pprint
			# pprint(vars(node))
			# raise Exception(node)
		node.children = []
		for child in node.get_children():
			node.children.append(child)
			self.prepare(child, sourceText)

	def show(self, start=None, stop=None):
		if start is None and stop is None:
			return self.text
		else:
			return self.text[start - self._start_offset:stop - self._start_offset]

	def hash(self):
		# TODO: Improve
		return self.name

	def diffLCS(self, other, mode):
		diff = []
		Y = self.node.text.split('\n')
		X = other.node.text.split('\n') if other is not None else []
		C = LCS(X, Y)
		getDiff(C, X, Y, len(X), len(Y), diff, mode)
		return diff


class Class(Object):
	def __init__(self, node, sourceFile):
		super(Class, self).__init__(node, sourceFile)


def simplehash(self):
	return self.hash

def eq(self, other):
	return self.hash == other.hash

def cmd(self, other):
	return self.hash > other.hash

ci.Cursor.__hash__ = simplehash

class Function(Object):

	def __init__(self, node, name, sourceFile, declarations, globals):
		super(Function, self).__init__(node, name, sourceFile)
		self.variables = dict()
		self.reversedVars = dict()
		self.declarations = declarations
		self.globals = globals
		self.parse()

	def parse(self):
		for nodeHash, node in self.globals.iteritems():
			self.variables[nodeHash] = list()
			self.reversedVars[nodeHash] = node

		self.__parse(self.node, [])


	def __parse(self, node, kind_path):
		kind_path_copy = list(kind_path)
		kind_path_copy.append(node.kind)

		node.__hash__ = types.MethodType(simplehash, node)
		node.__eq__ = types.MethodType(eq, node)
		node.__cmp__ = types.MethodType(cmp, node)

		if node.kind in [ci.CursorKind.VAR_DECL, ci.CursorKind.PARM_DECL]:
			self.variables[node.hash] = list()
			self.reversedVars[node.hash] = node
		elif node.kind == ci.CursorKind.DECL_REF_EXPR:
			definition = node.get_definition()
			if definition is not None:
				if self.variables.get(definition.hash, None) is not None:
					self.variables[definition.hash].append(kind_path_copy)
			elif self.declarations.get(node.displayname, None) is None or self.globals.get(node.displayname, None) is None:
					# print "UNDEFINED: %s" % (node.displayname)
					pass

		for c in node.children:
			self.__parse(c, kind_path_copy)

	def same(self, other) :
		logging.info("simpleDiff::%s x %s" % (type(self.node.text), type(other.node.text)))
		logging.info("simpleDiff::\nA:\n%s\n%s" % (':'.join(x.encode('hex') for x in self.node.text), ':'.join(x.encode('hex') for x in other.node.text)))
		logging.info("simpleDiff:: DECISION: %s" % (self.node.text == other.node.text))
		return self.node.text == other.node.text
		
	def structuralDiff(self, other, mode, changed=False):
		if mode == Mode.Old and other is not None and changed is False:
			return other.structuralDiff(self, mode, True)

		result = []
		for char in self.node.text:
			result.append((Color.Same, char))

		if other is not None:
			self.matchVars2(other)

		if other is not None:
			self.diff(self.node, other.node, mode, result)
		else:
			self.color(self.node, mode, result)
		return result


	def color(self, node, mode, result, start=0, stop=0):
		if len(node.text) > 0:
			for i in xrange(node.extent.start.offset+start, node.extent.end.offset-stop):
				char = node.text[i - node.extent.start.offset]
				result[i - self._start_offset] = (ModeToColor[mode], char)


	def matchVars2(self, other):
		matched = dict()
		matchingList = list()
		for var, positions in self.variables.iteritems():
			for new, new_positions in other.variables.iteritems():
				if len(positions) == 0:
					continue
				score = comapreVarPositions(positions, new_positions)
				matchingList.append((score, var, new))

		for s, c, b in matchingList:
			logging.info("MATCH_VAR_PROP %s -> %s :: %s" % (self.reversedVars[c].displayname, other.reversedVars[b].displayname, s))

		matchingList = sorted(matchingList, key=touplekey, reverse=True)

		used = set()
		for e in matchingList:
			score, c, b = e
			if b in used:
				continue
			if c in matched:
				continue
			logging.info("MATCH_VAR_DEC %s -> %s :: %s" % (self.reversedVars[c].displayname, other.reversedVars[b].displayname, score))
			used.add(b)
			matched[c] = b

		for c in self.variables.iterkeys():
			if c not in matched:
				matched[c] = None

		self.variablesMatched = matched


	# assume same kind
	def diffStmts(self, node, other, mode, result):
		logging.info("diffStmt :: (%s :: %s) -- (%s :: %s)" % (node.kind, node.text, other.kind, other.text))
		if node.kind in [ci.CursorKind.COMPOUND_STMT, ci.CursorKind.IF_STMT, ci.CursorKind.DECL_STMT]:
			for a, b in matchStmts2(node, other).iteritems():
				if b is None:
					self.color(a, mode, result)
				else:
					self.diff(a, b, mode, result)

		elif node.kind in [ci.CursorKind.CALL_EXPR]:
			if node.displayname != other.displayname:
				self.color(node, mode, result)
			else:
				self.diffChildren(node, other, mode, result)
			
		else:
			self.diffChildren(node, other, mode, result)

	def diffChildren(self, node, other, mode, result):
		z = zip(node.children, other.children)
		for a, b in z:
			self.diff(a, b, mode, result)
		for i in xrange(len(z), len(node.children)):
			self.color(node.children[i], mode, result)

	def diff(self, node, other, mode, result):
		if node.kind != other.kind:
			logging.info("diff :: (%s :: %s) -- (%s :: %s)" % (node.kind, node.text, other.kind, other.text))
			self.color(node, mode, result)
			return
		if ci.CursorKind.is_expression(node.kind):
			self.diffExpr(node, other, mode, result)
		elif ci.CursorKind.is_statement(node.kind):
			self.diffStmts(node, other, mode, result)
		elif ci.CursorKind.is_declaration(node.kind):
			self.diffDecl(node, other, mode, result)
		else:
			self.diffAny(node, other, mode, result)


	def diffExpr(self, node, other, mode, result):
		logging.info("diffExpr :: (%s :: %s) -- (%s :: %s)" % (node.kind, node.text, other.kind, other.text))
		if node.kind in [ci.CursorKind.DECL_REF_EXPR]:
			if node.displayname != other.displayname:
				definition = node.get_definition()
				definition_o = other.get_definition()
				if definition is not None and definition_o is not None:
					if self.variablesMatched.get(definition.hash, None) == definition_o.hash:
						self.color(node, Mode.Both, result)
					else:
						logging.info("Eliminated by not found (%s %s), (%s %s)" % (definition.displayname, definition.hash, definition_o.displayname, definition_o.hash))
						self.color(node, mode, result)
				else:
					logging.info("Eliminated by None %s, %s" % (node.displayname, other.displayname))
					self.color(node, mode, result)
		elif node.kind in [
			ci.CursorKind.INTEGER_LITERAL, ci.CursorKind.INTEGER_LITERAL, ci.CursorKind.FLOATING_LITERAL,
			ci.CursorKind.IMAGINARY_LITERAL, ci.CursorKind.STRING_LITERAL, ci.CursorKind.CHARACTER_LITERAL]:
			if node.text != other.text:
				self.color(node, mode, result)
		elif node.kind in [ci.CursorKind.BINARY_OPERATOR]:
			operator_a = node.text.rstrip(node.children[1].text).lstrip(node.children[0].text).strip()
			operator_b = other.text.rstrip(other.children[1].text).lstrip(other.children[0].text).strip()
			if operator_a != operator_b:
				self.color(node, mode, result, start=len(node.children[0].text), stop=len(node.children[1].text))
			for a, b in zip(node.children, other.children):
				self.diff(a, b, mode, result)
		elif node.kind in [ci.CursorKind.CALL_EXPR]:
			if node.displayname != other.displayname:
				self.color(node, mode, result)
			else:
				self.diffChildren(node, other, mode, result)
		else:
			self.diffChildren(node, other, mode, result)

	def diffDecl(self, node, other, mode, result):
		logging.info("diffDecl :: (%s :: %s) -- (%s :: %s)" % (node.kind, node.text, other.kind, other.text))
		if node.kind in [ci.CursorKind.VAR_DECL]:
			if node.displayname != other.displayname:
				if self.variablesMatched.get(node.hash, None) == other.hash:
					self.color(node, Mode.Both, result)
				else:
					self.color(node, mode, result)
			self.diffChildren(node, other, mode, result)
		else:
			self.diffChildren(node, other, mode, result)

	def diffAny(self, node, other, mode, result):
		logging.info("diffAny :: (%s :: %s) -- (%s :: %s)" % (node.kind, node.text, other.kind, other.text))
		for a, b in zip(node.children, other.children):
			self.diff(a, b, mode, result)


def compareTouples(t1, t2):
	s1, c1, b1 = t1
	s2, c2, b2 = t2
	return s1 < s2

def touplekey(t):
	s, _, _ = t
	return s

def matchStmts2(node, other):
	matched = dict()
	matchingList = list()
	for c in node.children:
		c_size = getSize(c)
		for b in other.children:
			if c.kind != b.kind:
				continue
			b_size = getSize(b)
			score = compareStruct(c, b)
			score = score / (c_size + b_size - score)
			matchingList.append((score, c, b))

	for s, c, b in matchingList:
		logging.info("MATCH_PROP %s -> %s :: %s" % (c.text, b.text, s))

	matchingList = sorted(matchingList, key=touplekey, reverse=True)

	used = set()
	for e in matchingList:
		score, c, b = e
		if b in used:
			continue
		if c in matched:
			continue
		logging.info("MATCH_DEC %s -> %s :: %s" % (c.text, b.text, score))
		used.add(b)
		matched[c] = b

	for c in node.children:
		if c not in matched:
			matched[c] = None

	return matched

# TODO: improve scoring
def compareStruct(a, b, debug=False):
	if debug:
		logging.info("compare (%s :: %s) with (%s :: %s)" %(a.kind, a.text, b.kind, b.text))
		logging.info("%s" % a.children)
	totalScore = 0.0
	if a.kind != b.kind:
		if debug:
			logging.info("%s, %s" % (a.kind, b.kind))
		return totalScore
	if a.displayname != "":
		totalScore += 0.5
		if a.displayname == b.displayname:
			totalScore += 0.5
	else:
		totalScore += 1.0

	if a.kind == ci.CursorKind.COMPOUND_STMT:
		matched = matchStmts2(a, b)
		for a, b in matched.iteritems():
			if b is not None:
				totalScore += compareStruct(a, b)
	else:
		for c, c_b in zip(a.children, b.children):
			totalScore += compareStruct(c, c_b, debug)

	return totalScore


def comapreVarPositions(a, b):
	score = 0.0
	for a_elem in a:
		for b_elem in b:
			partialScore = 0.0
			import difflib

			partialScore = difflib.SequenceMatcher(None, a_elem, b_elem)

			score += partialScore.ratio()

	if len(a) == 0 or len(b) == 0:
		return 0.0

	return score / ((len(a) * len(b)))


def getSize(node):
	s = 1
	for c in node.children:
		s += getSize(c)
	return s

# Parsers

class GitParser(object):

	__workFilePath = '/tmp/GDIworkFile%s'

	def __init__(self, repoPath):
		self.repoPath = repoPath
		self.repo = git.Repo(repoPath)


	def collectObjects(self, revision, last):
		self.repo.git.checkout(revision)
		# if len(self.repo.head.commit.parents) > 0 and revision.name_rev.split()[0] != last:
		# 	for diff in self.repo.head.commit.tree.diff(self.repo.head.commit.parents[0]):
		# 		print diff, diff.deleted_file, diff.a_path, diff.b_path, diff.new_file
		# 		if diff.new_file:
		# 			continue
		# 		basename, ext = os.path.splitext(self.repo.working_dir + '/' + diff.a_path)
		# 		if ext in ['.C', '.cpp', '.cp', '.c', '.cxx', '.cc', '.c++', '.h', '.hpp', '.hxx']:
		# 			yield self.repo.working_dir + '/' + diff.a_path
		# else:
		for i in self.repo.tree().traverse():
			basename, ext = os.path.splitext(i.abspath)
			if ext in ['.C', '.cpp', '.cp', '.c', '.cxx', '.cc', '.c++', '.h', '.hpp', '.hxx']:
				yield i.abspath

	def getRevisions(self, branch, last):
		head = self.repo.commit(branch)
		if last is not None:
			last = self.repo.commit(last)
		visited = set()
		queue = Queue.Queue()
		queue.put([head])
		history = []
		while not queue.empty():
			path = queue.get()
			rev = path[-1]
			if rev in visited:
				continue
			visited.add(rev)
			if rev == last:
				return reversed(path)
			for parent in rev.parents:
				new_path = path[:]
				new_path.append(parent)
				queue.put(new_path)
			history = path
		return reversed(history)

	def getSHA(self, rev):
		return self.repo.commit(rev)

class CParser(object):

	def __init__(self, filePath):
		self.filePath = filePath
		self.file = open(filePath, 'r')
		self.functions = []
		self.namespaces = []
		self.classes = []
		self.declarations = dict()
		self.globals = dict()

	def parse(self):
		index = ci.Index.create()
		self.filetext = self.file.read()
		tu = index.parse(self.filePath, args=['-std=c++11'])
		self.traverse(tu.cursor)
		self.file.close()
		return self.functions, self.classes
		
	def traverse(self, node, kind_path=[], name_prefix="", debug=False):
		kind_path_copy = list(kind_path)
		kind_path_copy.append(node.kind)
		if debug:
			print "\t" * len(kind_path), node.kind, node.displayname
		if node.kind in [ci.CursorKind.FUNCTION_DECL, ci.CursorKind.CONSTRUCTOR]:
			if node.is_definition() and os.path.abspath(self.filePath) == os.path.abspath(node.extent.start.file.name):
				self.functions.append(Function(node, name_prefix + node.displayname, self.filetext, self.declarations, self.globals))
				return
			else:
				self.declarations[node.displayname] = node
				return
		elif node.kind in [ci.CursorKind.CXX_METHOD]:
			if node.is_definition() and os.path.abspath(self.filePath) == os.path.abspath(node.extent.start.file.name):
				self.functions.append(Function(node, name_prefix + node.semantic_parent.displayname + "::" + node.displayname, self.filetext, self.declarations, self.globals))
				return
			else:
				self.declarations[node.displayname] = node
				return
		elif node.kind in [ci.CursorKind.NAMESPACE]:
			if node.displayname == "std" or node.displayname.startswith("_"):
				return
			name_prefix = name_prefix + node.displayname + "::"
		elif node.kind in [ci.CursorKind.VAR_DECL]:
			self.globals[node.hash] = node

		for children in node.get_children():
			self.traverse(children, kind_path_copy, name_prefix, debug)


class History(object):

	class Element(object):

		def __init__(self, function, revision):
			self.function = function
			self.revision = revision
			self.parents = []
			self.children = []

		def setChild(self, child):
			self.children.append(child)

		def setParent(self, parent):
			self.parents.append(parent)

		def __repr__(self):
			return "%s:%s\n" % (self.revision, self.function.name)

	def __init__(self, functionName):
		self.function = functionName
		self.head = None
		self.revisions = dict()

	def insert(self, function, revision):
		if revision.hexsha not in self.revisions:
			elem = History.Element(function, revision)
			self.revisions[revision.hexsha] = elem
			for parent in revision.parents:
				parent_elem = self.revisions.get(parent.hexsha, None)
				if parent_elem is not None:
					parent_elem.setChild(elem)
					elem.setParent(parent_elem)

	def setHead(self, head_rev):
		if head_rev.hexsha in self.revisions:
			self.head = self.revisions[head_rev.hexsha]
		else:
			print("no head", head_rev, self.function.name)
			# raise Exception("no head", head_rev, self.revisions)


	def clean(self):
		visited = set()
		queue = Queue.Queue()
		queue.put(self.head)
		while not queue.empty():
			rev = queue.get()
			if rev in visited:
				continue
			visited.add(rev)
			newChildren = list()
			for child in rev.children:
				if child in visited:
					newChildren.append(child)
			rev.children = newChildren
			for parent in rev.parents:
				queue.put(parent)

	def removeNoChanges(self):
		visited = set()
		queue = Queue.Queue()
		queue.put(self.head)
		while not queue.empty():
			rev = queue.get()
			if rev in visited:
				continue
			visited.add(rev)
			logging.info("Visit:%s" % rev.function.node.text)
			for child in rev.children:
				logging.info("Child:%s" % child.function.node.text)
				if child.function.same(rev.function):
					logging.info("Dropping")
					child.parents = [elem for elem in child.parents if elem != rev]
					child.parents += rev.parents
					for parent in rev.parents:
						parent.children = [elem for elem in parent.children if elem != rev]
						parent.children.append(child)
			for parent in rev.parents:
				queue.put(parent)


	def getRev(self, revision):
		if revision in self.tree:
			return self.revisions[revision]
		return None

	def is_single(self):
		return len(self.head.parents) == 0

	def __repr__(self):
		string = ""
		elem = self.head
		while elem is not None:
			string += str(elem)
			if len(elem.parents) > 0:
				elem = elem.parents[0]
			else:
				elem = None
		return string

class History2(object):

	class Element(object):

		def __init__(self, function, revision):
			self.function = function
			self.revision = revision
			self.parents = []
			self.children = []

		def setChild(self, child):
			self.children.append(child)

		def setParent(self, parent):
			self.parents.append(parent)

		def __repr__(self):
			return "%s:%s\n" % (self.revision, self.function.name)

	def __init__(self, functionName):
		self.function = functionName
		self.head = None
		self.revisions = set()
		self.last = None
		self.changed = False

	def insert(self, function, revision):
		# print function.name
		if revision.hexsha not in self.revisions:
			self.revisions.add(revision.hexsha)
			if self.last is None or not function.same(self.last.function):
				# print('adding', function.name)
				elem = History2.Element(function, revision)
				if self.last is not None:
					elem.setParent(self.last)
					self.last.setChild(elem)
				self.last = elem
			self.changed = True
		else:
			print 'WARN:', 'adding same revision'
			print function.name, self.last.function.name

	def setHead(self, head_rev):
		self.head = self.last

	def commit(self):
		self.changed = False

	def clean(self):
		visited = set()
		queue = Queue.Queue()
		queue.put(self.head)
		while not queue.empty():
			rev = queue.get()
			if rev in visited:
				continue
			visited.add(rev)
			newChildren = list()
			for child in rev.children:
				if child in visited:
					newChildren.append(child)
			rev.children = newChildren
			for parent in rev.parents:
				queue.put(parent)

	def removeNoChanges(self):
		visited = set()
		queue = Queue.Queue()
		queue.put(self.head)
		while not queue.empty():
			rev = queue.get()
			if rev in visited:
				continue
			visited.add(rev)
			logging.info("Visit:%s" % rev.function.node.text)
			for child in rev.children:
				logging.info("Child:%s" % child.function.node.text)
				if child.function.same(rev.function):
					logging.info("Dropping")
					child.parents = [elem for elem in child.parents if elem != rev]
					child.parents += rev.parents
					for parent in rev.parents:
						parent.children = [elem for elem in parent.children if elem != rev]
						parent.children.append(child)
			for parent in rev.parents:
				queue.put(parent)


	def getRev(self, revision):
		if revision in self.tree:
			return self.revisions[revision]
		return None

	def is_single(self):
		return len(self.head.parents) == 0

	def __repr__(self):
		string = ""
		elem = self.head
		while elem is not None:
			string += str(elem)
			if len(elem.parents) > 0:
				elem = elem.parents[0]
			else:
				elem = None
		return string



class Storage(object):
	def __init__(self):
		self.data = dict()
		self.ordered_data = dict()

	def add(self, func, revision):
		if func.hash() not in self.data:
			self.data[func.hash()] = History2(func)
		self.data[func.hash()].insert(func, revision)

	def checkRemoved(self, revision):
		for name, func in self.data.items():
			if not func.changed:
				print 'Removing', name
				del self.data[name]

	def clean(self, sha):
		for func in self.data.itervalues():
			func.setHead(sha)
			# if func.head is not None:
			# 	func.clean()
				# func.removeNoChanges()
		result = []
		for k, v in self.data.items():
			if v.head is not None and not v.is_single():
				result.append((k, v))
		self.ordered_data = collections.OrderedDict(sorted(result, key=lambda t: t[0]))


def getArgs():
	parser = argparse.ArgumentParser(description='Git Diff Improved')
	parser.add_argument('path', metavar='path', help='path to repo with c/c++ code')
	parser.add_argument('start_rev', metavar='revision', help='commit to start with (newest)')
	parser.add_argument('--last', '-l', default=None, help='last commit, initial commit is default')
	parser.add_argument('--mode', '-m', default='struct')
	parser.add_argument('--stats', action='store_true', default=False)
	parser.add_argument('--phase', '-p')

	return parser.parse_args()

# execute

def createStore(path, start_rev='master', last=None):
	storage = Storage()
	parser = GitParser(path)

	rev_sha = parser.getSHA(start_rev)

	count = 0
	for revision in parser.getRevisions(start_rev, last):
		count += 1
		print revision
		for filePath in parser.collectObjects(revision, last):
			functions, classes = CParser(filePath).parse()
			
			for function in functions:
				storage.add(function, revision)

		storage.checkRemoved(revision)
	storage.clean(rev_sha)

	return storage, count


mode = None

def main():
	init()
	args = getArgs()
	global mode
	mode = args.mode
	storage, count = createStore(args.path, args.start_rev, last=args.last)
	if args.stats:
		stats.gather_stats(storage, args.path.split("/")[-1], args.start_rev, args.last, args.phase, count)
		return

	ui.run(storage, args.mode)


if __name__ == '__main__':
	main()
