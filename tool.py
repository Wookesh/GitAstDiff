#!/usr/bin/env python
import git
import clang.cindex
import argparse
from ctypes.util import find_library
import os
import utils

clang.cindex.Config.set_library_file(find_library('clang'))


class Object(object):

	def __init__(self, node, sourceFile):
		self.node = node
		self.name = node.displayname
		self._start_offset = node.extent.start.offset
		sourceFile.seek(node.extent.start.offset)
		self.text = sourceFile.read(node.extent.end.offset - node.extent.start.offset)

	def show(self, start=None, stop=None):
		if start is None and stop is None:
			return self.text
		else:
			return self.text[start - self._start_offset:stop - self._start_offset]

	def hash(self):
		# TODO: Improve
		return self.name


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
					workFile.write(self.repo.git.show('%s:%s' % (revision, originalFilePath)))
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
		return history


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
		
	def traverse(self, node, indent=""):
		if node.kind in [clang.cindex.CursorKind.FUNCTION_DECL, clang.cindex.CursorKind.CONSTRUCTOR, clang.cindex.CursorKind.CXX_METHOD]:
			self.functions.append(Function(node, self.file))
		# elif node.kind == clang.cindex.CursorKind.CLASS_DECL:
		# 	self.classes.append(Function(node, self.file))
		for c in node.get_children():
			self.traverse(c, indent+"\t")


class History(object):

	class Element(object):

		def __init__(self, function, revision, following=None):
			self.function = function
			self.revision = revision
			self.following = following

		def next(self):
			return self.following

		def __repr__(self):
			return "%s:%s\n" % (self.revision, self.function.name)

	def __init__(self, functionName):
		self.function = functionName
		self.head = None
		self.data = dict()

	def insert(self, function, revision, after=None):
		if revision not in self.data:
			elem = History.Element(function, revision, self.head)
			self.data[revision] = elem
			self.head = elem
		else:
			raise Exception('failed when trying to add %s:%s for %s' % (revision, self.data, self.function))

	def getRev(self, revision):
		if revision in tree:
			return self.data[revision]
		return None

	def __repr__(self):
		string = ""
		elem = self.head
		while elem is not None:
			string += str(elem)
			elem = elem.following
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

def main():
	storage = Storage()
	args = getArgs()
	parser = GitParser(args.path)

	for revision in parser.getRevisions(args.branch):
		for file, tmpFile in parser.collectObjects(revision):
			print "Parsing %s" % file
			functions, classes = CParser(tmpFile).parse()
			
			for function in functions:
				print function.show()
				storage.add(function, revision)

	print storage.data

if __name__ == '__main__':
	main()
