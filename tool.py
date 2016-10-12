#!/usr/bin/env python
import git
import clang.cindex
import argparse
from ctypes.util import find_library
import os
import ui

clang.cindex.Config.set_library_file(find_library('clang-3.8'))

class Color(object):
	New	 = 0
	Changed = 1
	Same	= 2
	Removed = 3
	Differ  = 4


def findDiff(old, new):

	new_list = list(old.get_children())
	old_list = list(new.get_children())
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



def getDiff(C, X, Y, i, j, diff):
	if i > 0 and j > 0 and X[i-1] == Y[j-1]:
		getDiff(C, X, Y, i-1, j-1, diff)
		diff.append((0, X[i-1]))
	else:
		if j > 0 and (i == 0 or C[i][j-1] >= C[i-1][j]):
			getDiff(C, X, Y, i, j-1, diff)
			diff.append((2, Y[j-1]))
		elif i > 0 and (j == 0 or C[i][j-1] < C[i-1][j]):
			getDiff(C, X, Y, i-1, j, diff)
			diff.append((1, X[i-1]))


def desc(node, tab=""):
	print tab, node.kind
	for i in node.get_children():
		desc(i, tab+"\t")


class Object(object):

	def __init__(self, node, sourceFile):
		self.node = node
		self.name = node.displayname
		self._start_offset = node.extent.start.offset
		sourceFile.seek(node.extent.start.offset)
		# self._deepTextCopy(node, sourceFile.read())
		self.text = sourceFile.read(node.extent.end.offset - node.extent.start.offset)

	def _deepTextCopy(self, node, sourceText):
		node.text = sourceText[node.extent.start.offset:node.extent.end.offset]
		for child in node.get_children():
			self._deepTextCopy(child, sourceText)

	def show(self, start=None, stop=None):
		if start is None and stop is None:
			return self.text
		else:
			return self.text[start - self._start_offset:stop - self._start_offset]

	def hash(self):
		# TODO: Improve
		return self.name

	def diff(self, other):
		diff = []
		Y = self.text.split('\n')
		X = other.text.split('\n') if other is not None else []
		C = LCS(X, Y)
		getDiff(C, X, Y, len(X), len(Y), diff)
		return diff

class Class(Object):

	def __init__(self, node, sourceFile):
		super(Function, self).__init__(node, sourceFile)


class Function(Object):

	def __init__(self, node, sourceFile):
		super(Function, self).__init__(node, sourceFile)

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
		self.classes = []

	def parse(self):
		index = clang.cindex.Index.create()
		tu = index.parse(self.filePath)
		self.traverse(tu.cursor)
		self.file.close()
		return self.functions, self.classes
		
	def traverse(self, node, ident=""):
		if node.kind in [clang.cindex.CursorKind.FUNCTION_DECL, clang.cindex.CursorKind.CONSTRUCTOR, clang.cindex.CursorKind.CXX_METHOD] and node.is_definition():
			self.functions.append(Function(node, self.file))
		# elif node.kind == clang.cindex.CursorKind.CLASS_DECL:
		# 	self.classes.append(Function(node, self.file))
		for children in node.get_children():
			self.traverse(children, ident+"\t")


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
	args = parser.parse_args()
	return args

# execute

def createStore(path, branch):
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
	args = getArgs()
	storage = createStore(args.path, args.branch)

	ui.run(storage)


if __name__ == '__main__':
	main()
