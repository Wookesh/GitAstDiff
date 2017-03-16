#!/usr/bin/env python
import git
import clang.cindex
import argparse
from ctypes.util import find_library
import os
import ui
import logging

def init():
	logging.basicConfig(filename="tool.log", level=logging.DEBUG)
	clang.cindex.Config.set_library_file(find_library('clang-3.8'))
	logging.info("test")

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
	Mode.Both: Color.Same,
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


def findDiff(old, new):

	new_list = list(old.children)
	old_list = list(new.children)
	new_set = set(( e.hash for e in new_list))
	old_set = set(( e.hash for e in old_list))

	removed_elems = [e for e in old_list if e not in new_set]
	new_elems = [e for e in new_list if e not in old_set]

	if len(removed_elems) > 0:
		print removed_elems

	if len(new_elems) > 0:
		print new_elems


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
		# sourceFile.seek(self._start_offset)
		self.prepare(self.node, sourceFile.read())
		# self.text = sourceFile.read(node.extent.end.offset - self._start_offset)

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

	def diff(self, other, mode):
		diff = []
		Y = self.text.split('\n')
		X = other.text.split('\n') if other is not None else []
		C = LCS(X, Y)
		getDiff(C, X, Y, len(X), len(Y), diff, mode)
		return diff


class Class(Object):

	def __init__(self, node, sourceFile):
		super(Function, self).__init__(node, sourceFile)


class Function(Object):

	def __init__(self, node, sourceFile, declarations, globals):
		super(Function, self).__init__(node, sourceFile)
		self.variables = dict()
		self.declarations = declarations
		self.globals = globals
		self.parse()

	def parse(self):
		self.__parse(self.node, [self.node.kind])
		self.buildKindMap()
		# print "Variables:", self.variables
		# print "Globals:", self.globals
		# print "Declarations:", self.declarations
		

	def __parse2(self, node, kind_path):
		kind_path_copy = list(kind_path)
		kind_path_copy.append(node.kind)




	def __parse(self, node, kind_path):
		kind_path_copy = list(kind_path)
		kind_path_copy.append(node.kind)
		print "\t"*len(kind_path), node.kind, node.displayname

		if node.kind == clang.cindex.CursorKind.DECL_REF_EXPR:
			if self.variables.get(node.displayname, None) == None:
				if self.declarations.get(node.displayname, None) == None or self.globals.get(node.displayname, None) == None:
					# print "UNDEFINED VARIABLE: %s" % (node.displayname)
					pass
			else:
				self.variables[node.displayname].append(kind_path_copy)
		elif node.kind == clang.cindex.CursorKind.VAR_DECL:
			self.variables[node.displayname] = list()
		elif node.kind == clang.cindex.CursorKind.CALL_EXPR:
			pass


		for c in node.children:
			self.__parse(c, kind_path_copy)


	def buildKindMap(self):
		self.kindMap = dict()	
		def __buildKindMap(node, skip=False):
			if not skip:
				if self.kindMap.get(node.kind, None) == None:
					self.kindMap[node.kind] = list()
				self.kindMap[node.kind].append(node)
			for c in node.children:
				__buildKindMap(c)

		__buildKindMap(self.node, skip=True)
		print self.kindMap, "\n"

		
	def structuralDiff(self, other, mode, changed=False):
		if mode == Mode.Old and other is not None and changed is False:
			return other.structuralDiff(self, mode, True)

		result = []
		for char in self.node.text:
			result.append((Color.Same, char))

		if other is not None:
			self.__structuralDiff(self.node, other, other.node, mode, result, 0)
		return result

	def __structuralDiff(self, node, other, o_node, mode, result, depth):
		logging.info("%s%s, %s, %s" % ("\t"*depth, node.kind, o_node.kind, node.text))
		if node.kind != o_node.kind or node.displayname != o_node.displayname:
			logging.info("%s -- %s" % (node.displayname, o_node.displayname))
			if len(node.text) > 0:
				for i in xrange(node.extent.start.offset, node.extent.end.offset):
					logging.info("%s, %s, %s" % (i, node.extent.start.offset, len(node.text)))
					char = node.text[i - node.extent.start.offset]
					result[i - self._start_offset] = (ModeToColor[mode], char)
		else:
			# for n in node.children:
			# 	o = findBestMatching(other.kindMap, n)
			# 	logging.info("Proposing:%s" % (o))
			# 	if o is not None:
			# 		self.__structuralDiff(n, other, o, mode, result, depth+1)
			# compareBlock(node, o_node)

			for n1, n2 in zip(node.children, o_node.children):
				self.__structuralDiff(n1, other, n2, mode, result, depth+1)


def findBestMatching(kindMap, node):
	best = None
	bestScore = 0
	for t in kindMap[node.kind]:
		score = compare(t, node)
		if score > bestScore:
			bestScore = score
			best = t

	return best

def compareStruct(a, b):
	totalScore = 0.0
	if a.kind != b.kind:
		return totalScore
	totalScore += 1.0
	for c in a.children:
		bestScore = 0.0
		for c_b in b.children:
			if c.kind == c_b.kind:
				score = compareStruct(c, c_b)
				if score > bestScore:
					bestScore = score
				totalScore += score
	return totalScore
	

def compareBlock(a, b):
	if a.kind != b.kind or a.kind != clang.cindex.CursorKind.COMPOUND_STMT:
		return 0.0

	matchedStmt = dict()
	ids = dict()
	for stmt in a.children:
		ids[stmt.hash] = stmt
		matchedStmt[stmt.hash] = dict()
		for o_stmt in b.children:
			ids[o_stmt.hash] = o_stmt
			score = compareStmts(stmt, o_stmt)
			matchedStmt[stmt.hash][o_stmt.hash] = score
			if score == 100.0:
				break

	for stmt, matched in matchedStmt.iteritems():
		for o_stmt, score in matched.iteritems():
			logging.info("MATCHED:%s - %s : %s" % (ids[stmt].text, ids[o_stmt].text, score))


def compareStmts(a, b):
	if a.kind != b.kind:
		return 0.0
	if a.text == b.text:
		return 100.0
	return 50.0

def comapreForStmts(a, b):
	for a_aprt, b_part in zip(a.children, b.children):
		pass



# Parsers

class GitParser(object):

	__workFilePath = '/tmp/GDIworkFile%s'

	def __init__(self, repoPath):
		self.repoPath = repoPath
		self.repo = git.Repo(repoPath)

	def collectObjects(self, revision):
		for i in self.repo.commit(revision).tree.traverse():
			originalFilePath = os.path.relpath(i.abspath, self.repoPath)
			basename, ext = os.path.splitext(originalFilePath)
			if ext in ['.cpp', '.c', '.cxx']:
				with open(GitParser.__workFilePath % ext, 'w') as workFile:
					workFile.write(self.repo.git.show('%s:%s' % (revision, originalFilePath)).encode('utf-8'))
				yield originalFilePath, GitParser.__workFilePath % ext

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
		index = clang.cindex.Index.create()
		tu = index.parse(self.filePath)
		self.traverse(tu.cursor)
		self.file.close()
		return self.functions, self.classes
		
	def traverse(self, node, kind_path=[]):
		# print "\t"*len(kind_path), node.kind, node.displayname
		kind_path_copy = list(kind_path)
		kind_path_copy.append(node.kind)
		if node.kind in [clang.cindex.CursorKind.FUNCTION_DECL, clang.cindex.CursorKind.CONSTRUCTOR, clang.cindex.CursorKind.CXX_METHOD]:
			if node.is_definition():
				self.functions.append(Function(node, self.file, self.declarations, self.globals))
				return
			else:
				self.declarations[node.displayname] = node
		# elif node.kind in [clang.cindex.CursorKind.NAMESPACE]:
		# 	print "\t"*len(kind_path), node.kind, node.displayname
		elif node.kind in [clang.cindex.CursorKind.VAR_DECL] and node.is_definition():
			# print ident, node.kind, node.displayname
			self.globals[node.displayname] = list(kind_path_copy)
		# elif node.kind == clang.cindex.CursorKind.CLASS_DECL:
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
			# print self.data
			# raise Exception('failed when trying to add %s:%s for %s' % (revision, self.data, self.function.name))

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

	return parser.parse_args()

# execute

def createStore(path, branch='master'):
	storage = Storage()
	parser = GitParser(path)
	for revision in parser.getRevisions(branch):
		for file, tmpFile in parser.collectObjects(revision):
			# print "Parsing %s" % file
			functions, classes = CParser(tmpFile).parse()
			
			for function in functions:
				# print function.show()
				storage.add(function, revision)
	return storage


def main():
	init()
	args = getArgs()
	storage = createStore(args.path, args.branch)

	ui.run(storage)


if __name__ == '__main__':
	main()
