#!/usr/bin/python
import git

class GitParser(object):

	def __init__(self, repoPath):
		self.repo = git.Repo(repoPath)


	def collectObjects(self, commit):
		files = []
		for i in self.repo.commit(commit).tree.traverse():
			files.append(str(i.name))
		
		return files


def main():

	repoPath = raw_input("Repository path: ").strip()

	parser = GitParser(repoPath)
	allFiles = parser.collectObjects('master')



if __name__ == '__main__':
	main()