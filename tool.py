#!/usr/bin/env python
import git
import clang.cindex as ci
import argparse
from ctypes.util import find_library
import os
import ui
import logging
import types

def init():
	logging.basicConfig(filename="tool.log", level=logging.DEBUG)
	ci.Config.set_library_file(find_library('clang-3.8'))

class Color(object):
	Same	= 0
	Removed = 1
	New	    = 2
	Marked  = 3
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

	def __init__(self, node, sourceFile):
		self.node = node
		self.name = node.displayname
		self._start_offset = node.extent.start.offset
		self.prepare(self.node, sourceFile)

	def prepare(self, node, sourceText):
		node.text = sourceText[node.extent.start.offset:node.extent.end.offset]
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
		super(Function, self).__init__(node, sourceFile)


def simplehash(self):
	return self.hash

def eq(self, other):
	return self.hash == other.hash

def cmd(self, other):
	return self.hash > other.hash

ci.Cursor.__hash__ = simplehash

class Function(Object):

	def __init__(self, node, sourceFile, declarations, globals):
		super(Function, self).__init__(node, sourceFile)
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
			elif self.declarations.get(node.displayname, None) == None or self.globals.get(node.displayname, None) == None:
					# print "UNDEFINED: %s" % (node.displayname)
					pass

		for c in node.children:
			self.__parse(c, kind_path_copy)

		
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



	def matchVariables(self, other):
		self.variablesMatched = dict()

		# for var in self.variables.iterkeys():
		# 	if var in other.variables:
		# 		self.variablesMatched[var] = var


		for var, positions in self.variables.iteritems():
			if var not in other.variables:
				possible = None
				possible_value = 0.0
				for new, new_positions in other.variables.iteritems():
					if new in self.variablesMatched:
						continue

					similarity_val = comapreVarPositions(positions, new_positions)

					if similarity_val > possible_value:
						possible = new
						possible_value = similarity_val
				if possible != None:
					self.variablesMatched[var] = possible
					logging.info("matched %s -> %s" % (self.reversedVars[var].displayname, other.reversedVars[possible].displayname))
				else:
					logging.info("cannot match: %s" % self.reversedVars[var].displayname)
			else:
				logging.info("matched direct %s -> %s" % (self.reversedVars[var].displayname, self.reversedVars[var].displayname))
				self.variablesMatched[var] = var

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


# TODO:
#       find nested
def matchStmts(node, other):
	matched = dict()
	for c in node.children:
		possible = dict()
		for b in other.children:
			if c.kind != b.kind:
				continue
			score = compareStruct(c, b)
			foundbetter = False
			for k, v in matched.iteritems():
				s = v.get(b, None)
				if s is not None:
					if s < score:
						v.pop(b)
					elif s > score:
						foundbetter = True
			if not foundbetter:
				possible[b] = score
		matched[c] = possible

	result = dict()
	for c, elems in matched.iteritems():
		best = None
		bestScore = 0.0
		for k, v in elems.iteritems():
			if v >= bestScore:
				bestScore = v
				best = k
		result[c] = best

	for k, v in result.iteritems():
		if v is not None:
			logging.info("%s::%s -> %s::%s" % (k.kind, k.text, v.kind, v.text))
		else:
			logging.info("%s::%s -> None" % (k.kind, k.text))

	return result


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
		# for c in a.children:
		# 	bestScore = 0.0
		# 	for c_b in b.children:
		# 		if c.kind == c_b.kind:
		# 			score = compareStruct(c, c_b)
		# 			if score > bestScore:
		# 				bestScore = score
		# 			totalScore += score
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
			
			# for i in xrange(0, min(len(a_elem), len(b_elem)) - 1):
			# 	if i > 2 and a_elem[i] == b_elem[i]:
			# 		partialScore += 1.0
			# partialScore = partialScore / (len(a_elem) +  len(b_elem) - partialScore)
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


	def collectObjects(self, revision):
		self.repo.git.checkout(revision)
		for i in self.repo.tree().traverse():
			basename, ext = os.path.splitext(i.abspath)
			if ext in ['.cpp', '.c', '.cxx']:
				yield i.abspath

	def getRevisions(self, branch):
		head = self.repo.commit(branch)
		history = list()
		visited = set()
		queue = [head]
		while queue:
			rev = queue.pop()
			if rev in visited:
				continue
			visited.add(rev)
			history.append(rev)
			queue += rev.parents
		return reversed(history)

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
		tu = index.parse(self.filePath)
		self.traverse(tu.cursor)
		self.file.close()
		return self.functions, self.classes
		
	def traverse(self, node, kind_path=[]):
		kind_path_copy = list(kind_path)
		kind_path_copy.append(node.kind)
		# print "\t" * len(kind_path), node.kind, node.displayname
		if node.kind in [ci.CursorKind.FUNCTION_DECL, ci.CursorKind.CONSTRUCTOR, ci.CursorKind.CXX_METHOD]:
			if node.is_definition():
				self.functions.append(Function(node, self.filetext, self.declarations, self.globals))
				return
			else:
				self.declarations[node.displayname] = node
		# elif node.kind in [ci.CursorKind.NAMESPACE]:
		# 	print "\t"*len(kind_path), node.kind, node.displayname
		elif node.kind in [ci.CursorKind.VAR_DECL]:
			self.globals[node.hash] = node
		# elif node.kind == ci.CursorKind.CLASS_DECL:
		# 	self.classes.append(Function(node, self.file))
		for children in node.get_children():
			self.traverse(children, kind_path_copy)


class History(object):

	class Element(object):

		def __init__(self, function, revision, parent=None, child=None):
			self.function = function
			self.revision = revision
			self.parent = parent
			self.child = child

		def setChild(self, child):
			self.child = child

		def __repr__(self):
			return "%s:%s\n" % (self.revision, self.function.name)

	def __init__(self, functionName):
		self.function = functionName
		self.head = None
		self.data = dict()

	def insert(self, function, revision, after=None):
		if revision not in self.data:
			elem = History.Element(function, revision, self.head)
			if self.head is not None:
				self.head.setChild(elem)
			self.data[revision] = elem
			self.head = elem
		else:
			pass

	def getRev(self, revision):
		if revision in self.tree:
			return self.data[revision]
		return None

	def __repr__(self):
		string = ""
		elem = self.head
		while elem is not None:
			string += str(elem)
			elem = elem.parent
		return string


class Storage(object):

	def __init__(self):
		self.data = dict()

	def add(self, function, revision):
		if function.hash() not in self.data:
			self.data[function.hash()] = History(function)
		self.data[function.hash()].insert(function, revision)


def getArgs():
	parser = argparse.ArgumentParser(description='Git Diff Improved')
	parser.add_argument('path', metavar='path', help='path to repo with c/c++ code')
	parser.add_argument('branch', metavar='branch', help='branch/tag name of repo to parse')
	parser.add_argument('--count', '-c', default=-1, type=int)
	parser.add_argument('--mode', '-m', default='struct')

	return parser.parse_args()

# execute

def createStore(path, branch='master', count=-1):
	storage = Storage()
	parser = GitParser(path)
	counter = 0
	for revision in parser.getRevisions(branch):
		for filePath in parser.collectObjects(revision):
			functions, classes = CParser(filePath).parse()
			
			for function in functions:
				storage.add(function, revision)
		counter += 1
		if counter == count:
			break
	return storage


def main():
	init()
	args = getArgs()
	storage = createStore(args.path, args.branch, count=args.count)

	ui.run(storage, args.mode)


if __name__ == '__main__':
	main()
