#!/usr/bin/env python
import git
import clang.cindex
from ctypes.util import find_library


class GitParser(object):

	def __init__(self, repoPath):
		self.repo = git.Repo(repoPath)


	def collectObjects(self, commit):
		files = []
		for i in self.repo.commit(commit).tree.traverse():
			files.append(str(i.name))
		
		return files


class CParser(object):

	def __init__(self):
		pass

	@classmethod
	def parse(cls, filePath):
		index = clang.cindex.Index.create()
		tu = index.parse(filePath)
		CParser.traverse(tu.cursor)
		
	@classmethod
	def traverse(cls, node, indent=""):
		print "%s%s, %s" %(indent, node.kind, node.displayname)
		for c in node.get_children():
			CParser.traverse(c, indent+"\t")


def main():
	clang.cindex.Config.set_library_file(find_library('clang'))
	repoPath = raw_input("Repository path: ").strip()

	parser = GitParser(repoPath)
	allFiles = parser.collectObjects('master')

	for file in allFiles:
		if ".cpp" in file or ".c" in file:
			CParser.parse(file)



if __name__ == '__main__':
	main()