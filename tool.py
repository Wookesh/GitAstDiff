#!/usr/bin/python
import git

class Parser(object):

	def __init__(self, repoPath):
		self.repo = git.Repo(repoPath)


	def collectObjects(repoPath):
		pass


def main():

	repoPath = raw_input("Repository path: ").strip()

	parser = Parser(repoPath)





if __name__ == '__main__':
	main()