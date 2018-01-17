# Solves a randomized 8-puzzle using A* algorithm with plug-in heuristics

import random
import math
from collections import deque
finishState = [[0, 1, 2],
               [3, 4, 5],
               [6, 7, 8]]

def index(item, seq):
    """Helper function that returns -1 for non-found index value of a seq"""
    try:
        return seq.index(item)
    except:
        return -1

class Puzzle8:

    def __init__(self):
        # heuristic value
        self._hval = 0
        # search depth of current instance
        self._depth = 0
        self.max_depth = 0
        # parent node in search path
        self._parent = None
        # the action (i.e., the direction of the move) that has applied to the parent node.
        self.cost = 0
        self.direction = None
        self.matrix = []
        for i in range(3):
            self.matrix.append(finishState[i][:])

    def __eq__(self, other):
        if self.__class__ != other.__class__:
            return False
        else:
            return self.matrix == other.matrix

    def __str__(self):
        res = ''
        for row in range(3):
            res += ' '.join(map(str, self.matrix[row]))
            res += '\r\n'
        return res

    def change_state(self, l):
        """ change the state of the current puzzle to l"""
        if len(l) != 9:
            print ('length of the list should be 9')
        self.matrix = []
        for i in range(3):
            self.matrix.append(l[i*3:i*3+3])


    def clone(self):
        """ Create a copy of the existing puzzle"""
        p = Puzzle8()
        for i in range(3):
            p.matrix[i] = self.matrix[i][:]
        return p

    def getLegalMoves(self):
        """ Returns the set of available moves as a list of tuples, each tuple contains the row and col position
        with which the free space (0) may be swapped """
        # get row and column of the empty piece
        row, col = self.find(0)
        free = []

        # find which pieces can move there
        if row > 0:
            free.append((row - 1, col, 'up'))
        if row < 2:
            free.append((row + 1, col, 'down'))
        if col > 0:
            free.append((row, col - 1, 'left'))
        if col < 2:
            free.append((row, col + 1, 'right' ))

        return free


    def generateMoves(self):
        """ Returns a set of puzzle objects that are successors of the current state   """
        free = self.getLegalMoves()
        zero = self.find(0)

        def swap_and_clone(a, b):
            p = self.clone()
            p.swap(a,b[0:2])
            p.direction = b[2] # up, down, left or right
            p._depth = self._depth + 1
            if p.max_depth < p._depth:
                p.max_depth = p._depth+1

            p._parent = self
            return p
        # map applies the function swap_and_clone to each tuple in free, returns a list of puzzle objects
        return map(lambda pair: swap_and_clone(zero, pair), free)

    def generateSolutionPath(self, path):
        """ construct the solution path by recursively following the pointers to the parent  """
        if self._parent == None:
            return path
        else:
            path.append(self)
            return self._parent.generateSolutionPath(path)



    def shuffle(self, step_count):
        """shuffles the puzzle by executing step_count number of random moves"""
        for i in range(step_count):
            row, col = self.find(0)
            free = self.getLegalMoves()
            target = random.choice(free)
            self.swap((row, col), target[0:2])
            row, col = target[0:2]

    def find(self, value):
        """returns the row, col coordinates of the specified value
           in the graph"""
        if value < 0 or value > 8:
            raise Exception("value out of range")

        for row in range(3):
            for col in range(3):
                if self.matrix[row][col] == value:
                    return row, col

    def peek(self, row, col):
        """returns the value at the specified row and column"""
        return self.matrix[row][col]

    def poke(self, row, col, value):
        """sets the value at the specified row and column"""
        self.matrix[row][col] = value

    def swap(self, pos_a, pos_b):
        """swaps values at the specified coordinates"""
        temp = self.peek(*pos_a)
        self.poke(pos_a[0], pos_a[1], self.peek(*pos_b))
        self.poke(pos_b[0], pos_b[1], temp)

    def isGoal(puzzle):
        """check if we reached  the goal state"""
        return puzzle.matrix == finishState

    def BFS(self):
        """Performs BFS for goal state"""
        frontier = [self]
        explored= {str(self):1}
        expanded = -1

        print("explored", len(explored)==0)
        while frontier:
            node = frontier.pop(0)
            neighbours = node.generateMoves()
            expanded += 1

            if node.isGoal() :
                cost=1
                for c in node.generateSolutionPath([]): cost += 1
                path=[c.direction for c in reversed(node.generateSolutionPath([]))]
                pa= "path_to_goal:"+str(path)
                co= "cost_of_path:"+ str(cost)
                exp="nodes_expanded:"+str(expanded)
                dep="search_depth:"+str(node._depth)
                maxD = "max_deep_search:"+ str(neighbour.max_depth)
                file = open("output.txt","w")
                file.write(str(pa)+"\n");
                file.write(str(co)+"\n");
                file.write(str(exp)+"\n");
                file.write(str(dep)+"\n");
                file.write(str(maxD) + "\n");
                file.close();

                print("path_to_goal",[c.direction for c in reversed(node.generateSolutionPath([]))])
                print ("cost_of_path", cost)
                print("nodes_expanded",expanded)
                print("search_depth",(node._depth ))
                print("max_deep_search", node.max_depth)
                return True

            for neighbour in neighbours:
                if str(neighbour) not in explored.keys() and neighbour not in frontier:
                    frontier.append(neighbour)

            explored[str(node)]=1



    def DFS(self):
        """Performs DFS for goal state"""
        frontier = deque()
        frontier.append(self)
        stack = {str(self):1}
        explored= {str(self):1}
        expanded = -1

        while frontier:
            node = frontier.pop()

            if node.isGoal()== True :
                cost = 0
                for c in node.generateSolutionPath([]): cost += 1
                path=[c.direction for c in reversed(node.generateSolutionPath([]))]
                pa= "path_to_goal:"+str(path)
                co= "cost_of_path:"+ str(cost)
                exp="nodes_expanded:"+str(expanded)
                dep="search_depth:"+str(node._depth)
                maxD = "max_deep_search:"+ str(node.max_depth)
                file = open("output.txt","w")
                file.write(str(pa)+"\n");
                file.write(str(co)+"\n");
                file.write(str(exp)+"\n");
                file.write(str(dep)+"\n");
                file.write(str(maxD) + "\n");
                file.close();

                print("path_to_goal",[c.direction for c in reversed(node.generateSolutionPath([]))])
                for c in (node.generateSolutionPath([])): cost += 1
                print ("cost_of_path", (cost))
                print("nodes_expanded",expanded)
                print("search_depth",(node._depth ))
                print("max_deep_search", node.max_depth)
                return True

            neighbours = node.generateMoves()
            liste=[]

            for neighbour in neighbours:

                if str(neighbour) not in explored.keys() and str(neighbour) not in stack.keys():

                    frontier.appendleft(neighbour)
                    stack[str(neighbour)]=1
                    expanded += 1

            explored[str(node)] = 1



    def Astarsearch(self, h):
        """Performs A* search for goal state.
        h(move) - heuristic function, returns an integer
        """

        openl = [self]
        closedl = []
        frontier = {str(self):1}
        expanded = 0

        while len(openl) > 0:
            node = openl.pop(0)

            if node.isGoal() :
                cost=0
                for c in node.generateSolutionPath([]): cost += 1
                path=[c.direction for c in reversed(node.generateSolutionPath([]))]
                pa= "path_to_goal:"+str(path)
                co= "cost_of_path:"+ str(cost)
                exp="nodes_expanded"+str(expanded)
                dep="search_depth"+str(node._depth-1 )
                maxD = "max_deep_search"+ str(node.max_depth)
                file = open("output.txt","w")
                file.write(str(pa)+"\n");
                file.write(str(co)+"\n");
                file.write(str(exp)+"\n");
                file.write(str(dep)+"\n");
                file.write(str(maxD) + "\n");
                file.close();

                print("path",[c.direction for c in reversed(node.generateSolutionPath([]))])
                print("cost", cost)
                print("nodes_expanded",expanded)
                print("search_depth",(node._depth-1 ))
                print("max_deep_search", node.max_depth-1)
                return True


            neighbours = node.generateMoves()
            idx_open = idx_closed = -1

            for neighbour in neighbours:
                if str(neighbour) not in frontier.keys()  and str(neighbour) not in frontier.keys():
                    expanded += 1
                    frontier[str(neighbour)]=1
                    # have we already seen this node?
                    idx_open = index(neighbour, openl)
                    idx_closed = index(neighbour, closedl)
                    hval = h(neighbour)
                    fval = hval + neighbour._depth


                    if idx_closed == -1 and idx_open == -1:
                        neighbour._hval = hval
                        openl.append(neighbour)

                    elif idx_open > -1:
                        copy = openl[idx_open]
                        if fval < copy._hval + copy._depth:
                            # copy neighbour's values over existing
                            copy._hval = hval
                            copy._parent = neighbour._parent
                            copy._depth = neighbour._depth
                    elif idx_closed > -1:
                        copy = closedl[idx_closed]
                        if fval < copy._hval + copy._depth:
                            neighbour._hval = hval
                            closedl.remove(copy)
                            openl.append(neighbour)


            closedl.append(node)


        # if finished state not found, return failure
        return False
def heur(puzzle, item_total_calc, total_calc):
    """
    Heuristic template that provides the current and target position for each number and the
    total function.

    """
    t = 0
    for row in range(3):
        for col in range(3):
            val = puzzle.peek(row, col) - 1
            target_col = val % 3
            target_row = val / 3

            if target_row < 0:
                target_row = 2

            t += item_total_calc(row, target_row, col, target_col)

    return total_calc(t)

def manhattan(puzzle):
        return heur(puzzle,
                    lambda r, tr, c, tc: abs(tr - r) + abs(tc - c),
                    lambda t : t)




def main():
    p = Puzzle8() # when we create the puzzle object, it's already in the goal state
    # p.shuffle(20) # that's why we shuffle to start from a random state which is 20 steps away from from the goal state
    # print(p)
    p.change_state([1,2,5,3,4,0,6,7,8])
    print(p)
    p.BFS()
    p.DFS()
    p.Astarsearch(manhattan)
    # print("solution", p.BFS())


if __name__ == "__main__":
    main()
