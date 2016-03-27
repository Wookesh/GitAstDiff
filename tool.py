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
		print tu




def main():

	# repoPath = raw_input("Repository path: ").strip()

	# parser = GitParser(repoPath)
	# allFiles = parser.collectObjects('master')


	clang.cindex.Config.set_library_file(find_library('clang'))
	CParser.parse("test.cpp")



if __name__ == '__main__':
	main()