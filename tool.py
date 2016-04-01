#!/usr/bin/env python
import git
import clang.cindex
import argparse
from ctypes.util import find_library
import os

class Object(object):

	def __init__(self, node, sourceFile):
		self.node = node
		sourceFile.seek(node.extent.start.offset)
		self.text = sourceFile.read(node.extent.end.offset - node.extent.start.offset)


class Class(object):

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
		elif node.kind == clang.cindex.CursorKind.CLASS_DECL:
			self.classes.append(Function(node, self.file))
		for c in node.get_children():
			self.traverse(c, indent+"\t")

def getArgs():
	parser = argparse.ArgumentParser(description='Git Diff Improved')
	parser.add_argument('path', metavar='path', help='path to repo with c/c++ code')
	parser.add_argument('branch', metavar='branch', help='branch/tag name of repo to parse')
	args = parser.parse_args()
	return args


# execute

def main():
	args = getArgs()
	clang.cindex.Config.set_library_file(find_library('clang'))
	parser = GitParser(args.path)

	for file, tmpFile in parser.collectObjects(args.branch):
		print "Parsing %s" % file
		functions, classes = CParser(tmpFile).parse()

		print "#########################################"
		print "Functions:"
		for fun in functions:
			print fun.text

		print "#########################################"
		print "Classes:"
		for cls in classes:
			print cls.text


if __name__ == '__main__':
	main()
