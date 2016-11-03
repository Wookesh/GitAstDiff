import curses
import time
import textwrap
import tool
import logging

logging.basicConfig(filename='tool.log',level=logging.DEBUG)

class MenuOption(object):

	def __init__(self, name, follow):
		self.name = name
		self.follow = follow

	def execute(self, parent):
		self.follow.run(parent)

class FunctionOption(MenuOption):

	def __init__(self, function, name, diffViewer):
		self.name = name
		self.function = function
		self.follow = diffViewer

	def execute(self, parent):
		self.follow.run(parent, self.function)


class UIObject(object):

	def __init__(self):
		pass

	def run(self):
		pass

	def show():
		stdscr.clear()
		stdscr.addstr(0, 0, "DEFAULT SHOW")


class DiffViewer(UIObject):

	def __init__(self):
		curses.start_color()
		curses.use_default_colors()
		for i in range(0, curses.COLORS):
			curses.init_pair(i, i, -1)
		self.functionHistory = None
		self.columnSize = 80
		self.position = None

		self.gitBarHeight = 10

		self.height, self.width = stdscr.getmaxyx()
		self.window1 = curses.newwin(self.height - self.gitBarHeight, self.width / 2, self.gitBarHeight, 0)
		self.window2 = curses.newwin(self.height - self.gitBarHeight, self.width / 2, self.gitBarHeight, self.width / 2)
		self.gitWindow = curses.newwin(self.gitBarHeight, self.width, 0, 0)

	def __show(self, window, function):
		window.clear()
		if function is not None:
			window.addstr(0, 0, str(function.revision))
			lines = []
			for line in function.function.show().split('\n'):
				lines += textwrap.wrap(line, self.width / 2)
			for i in xrange(0, min(len(lines), self.height - 1)):
				window.addstr(i + 1, 0, lines[i])
		window.refresh()

	def __show2(self, window, function, mode):
		window.clear()
		if function is not None and (mode == tool.Mode.New or function.parent is not None):
			parent = function.parent.function if function.parent is not None else None
			window.addstr(0, 0, str(self.position.revision if mode == tool.Mode.New else self.position.parent.revision))
			lines = []
			for color, line in function.function.diff(parent, mode):
				for l in textwrap.wrap(line, self.width):
					lines.append((color, l))
			for i in xrange(0, min(len(lines), self.height - self.gitBarHeight - 1)):
				color, line = lines[i]
				window.addstr(i + 1, 0, line, curses.color_pair(color))
		window.refresh()

	def __showGit(self, currentRevision):
		self.gitWindow.clear()
		mid = list()
		rev = self.functionHistory.head
		comment = currentRevision.revision.summary
		author = currentRevision.revision.author.name + " " + currentRevision.revision.author.email
		while rev.parent is not None:
			if rev == currentRevision:
				mid.append('O')
			else:
				mid.append('*')
			mid.append('-')
			rev = rev.parent
		if rev == currentRevision:
			mid.append('O')
		else:
			mid.append('*')
		string = ""
		for elem in reversed(mid):
			string += elem
		self.gitWindow.addstr(1, 0 , string)
		self.gitWindow.addstr(3, 0, comment)
		self.gitWindow.addstr(4, 0, author)
		self.gitWindow.refresh()

	def show(self):
		self.__show2(self.window1, self.position, tool.Mode.Old)
		self.__show2(self.window2, self.position, tool.Mode.New)
		self.__showGit(self.position)

	def reset(self):
		self.functionHistory = None
		self.positions = None

	def goToParent(self):
		if self.position.parent is not None:
			self.position = self.position.parent

	def goToChild(self):
		if self.position.child is not None:
			self.position = self.position.child

	def run(self, parent, function):
		self.functionHistory = function
		self.position = self.functionHistory.head
		stdscr.clear()
		stdscr.refresh()
		self.window1.refresh()
		self.window2.refresh()
		while True:
			self.show()
			c = stdscr.getch()
			if c == curses.KEY_RIGHT:
				self.goToChild()
			if c == curses.KEY_LEFT:
				self.goToParent()
			if c == ord('q'):
				self.reset()
				break	


class Menu(UIObject):

	def __init__(self, options):
		self.options = options
		self.select = 0
		self.height, self.width = stdscr.getmaxyx()

	def show(self):
		stdscr.clear()
		first = min(max(self.select - self.height / 2, 0), max(len(self.options) - self.height, 0))
		for i in xrange(0, min(self.height, len(self.options))):
			sign = "* "
			if first + i == self.select:
				sign = "->"
			stdscr.addstr(i, 0, "%s %s" % (sign, self.options[first + i].name))
	
	def down(self, move=1):
		if self.select + move < len(self.options):
			self.select += move


	def up(self, move=1):
		if self.select - move + 1 > 0:
			self.select -= move

	def execute(self):
		if self.options[self.select].follow is not None:
			self.options[self.select].execute(self)
		else:
			stdscr.clear()
			stdscr.addstr(0, 0, "ERROR")
			stdscr.refresh()
			time.sleep(1)
			self.show()

	def run(self, parent = None):
		while True:
			self.show()
			c = stdscr.getch()
			if c == ord('q'):
				break
			elif c == curses.KEY_DOWN:
				self.down()
			elif c == curses.KEY_UP:
				self.up()
			elif c == curses.KEY_PPAGE:
				self.up(self.height)
			elif c == curses.KEY_NPAGE:
				self.down(self.height)
			elif c in [curses.KEY_ENTER, ord('\n')]:
				self.execute()


stdscr = None

def run(storage):
	global stdscr
	stdscr = curses.initscr()
	curses.noecho()
	curses.cbreak()
	stdscr.keypad(1)

	try:
		diffViewer = DiffViewer()
		options = []
		for fName, function in storage.data.iteritems():
			options.append(FunctionOption(function, fName, diffViewer))
		MainMenu = Menu(options)
		MainMenu.run()
	except Exception:
		raise
	finally:
		curses.nocbreak(); stdscr.keypad(0); curses.echo()
		curses.endwin()
