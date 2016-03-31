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
	def __init__(self, repoPath):
		self.repo = git.Repo(repoPath)


	def collectObjects(self, commit):
		files = []
		for i in self.repo.commit(commit).tree.traverse():
			files.append(str(i.name))
		
		return files

class CParser(object):

	def __init__(self, filePath):
		self.filePath = filePath
		self.file = open(filePath)
		self.functions = []
		self.classes = []

	def parse(self):
		index = clang.cindex.Index.create()
		tu = index.parse(self.filePath)
		self.traverse(tu.cursor)
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
	return parser.parse_args()


# execute

def main():
	args = getArgs()
	clang.cindex.Config.set_library_file(find_library('clang'))
	parser = GitParser(args.path)
	allFiles = parser.collectObjects(args.branch)

	for file in allFiles:
		if ".cpp" in file or ".c" in file:
			print "Parsing %s" % file
			functions, classes = CParser(os.path.join(args.path, file)).parse()

			print "Functions:"
			for fun in functions:
				print fun.text

			print "Classes:"
			for cls in classes:
				print cls.text


if __name__ == '__main__':
	main()
