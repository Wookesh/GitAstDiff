import curses
import time

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
		self.functionHistory = None
		self.columnSize = 80
		self.position = None
		self.window1 = curses.newwin(40,80,0,0)
		self.window2 = curses.newwin(40,80,0,80)

	def __show(self, window, function):
		window.clear()
		if function is not None:
			window.addstr(0, 0, str(function.revision))
			window.addstr(1, 0, function.function.show())
		window.refresh()

	def show(self):
		self.__show(self.window1, self.position)
		self.__show(self.window2, self.position.parent)
		#self.__showGit()

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
	except Exception, e:
		raise e
	finally:
		curses.nocbreak(); stdscr.keypad(0); curses.echo()
		curses.endwin()