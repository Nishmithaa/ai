# Artificial_Intelligence<br>
**Tower of hanoi**
def TowerOfHanoi(n , source, destination, auxiliary):<br>
    if n==1:<br>
        print ("Move disk 1 from source",source,"to destination",destination)<br>
        return<br>
    TowerOfHanoi(n-1, source, auxiliary, destination)<br>
    print ("Move disk",n,"from source",source,"to destination",destination)<br>
    TowerOfHanoi(n-1, auxiliary, destination, source)<br>
         n = 4<br>
TowerOfHanoi(n,'A','B','C')<br>

**Output**<br>
Move disk 1 from source A to destination C<br>
Move disk 2 from source A to destination B<br>
Move disk 1 from source C to destination B<br>
Move disk 3 from source A to destination C<br>
Move disk 1 from source B to destination A<br>
Move disk 2 from source B to destination C<br>
Move disk 1 from source A to destination C<br>
Move disk 4 from source A to destination B<br>
Move disk 1 from source C to destination B<br>
Move disk 2 from source C to destination A<br>
Move disk 1 from source B to destination A<br>
Move disk 3 from source C to destination B<br>
Move disk 1 from source A to destination C<br>
Move disk 2 from source A to destination B<br>
Move disk 1 from source C to destination B<br>

**Best first search**<br>
from queue import PriorityQueue<br>
import matplotlib.pyplot as plt<br>
import networkx as nx<br>


def best_first_search(source, target, n):<br>
    visited = [0] * n<br>
    visited[source] = True<br>
    pq = PriorityQueue()<br>
    pq.put((0, source))<br>
    while pq.empty() == False:<br>
        u = pq.get()[1]<br>
        print(u, end=" ") # the path having lowest cost<br>
        if u == target:<br>
            break<br>
<br>
        for v, c in graph[u]:<br>
            if visited[v] == False:<br>
                visited[v] = True
                pq.put((c, v))<br>
    print()<br>
<br>
def addedge(x, y, cost):<br>
    graph[x].append((y, cost))<br>
    graph[y].append((x, cost))<br>
v = int(input("Enter the number of nodes: "))<br>
graph = [[] for i in range(v)] # undirected Graph<br>
e = int(input("Enter the number of edges: "))<br>
print("Enter the edges along with their weights:")<br>
for i in range(e):<br>
    x, y, z = list(map(int, input().split()))<br>
    addedge(x, y, z)<br>
source = int(input("Enter the Source Node: "))<br>
target = int(input("Enter the Target/Destination Node: "))<br>
print("\nPath: ", end = "")<br>
best_first_search(source, target, v)<br>
**Output**<br>
Enter the number of nodes: 4<br>
Enter the number of edges: 5<br>
Enter the edges along with their weights:<br>
0 1 1<br>
0 2 1<br>
0 3 2 <br>
2 3 2<br>
1 3 3<br>
Enter the Source Node: 2<br>
Enter the Target/Destination Node: 1<br>

Path: 2 0 1 <br>

**Breadth first search**<br><br>
graph = {<br><br>
    '1' : ['2','10'],<br><br>
    '2' : ['3','8'],<br><br>
    '3' : ['4'],<br><br>
    '4' : ['5','6','7'],<br><br>
    '5' : [],<br><br>
    '6' : [],<br><br>
    '7' : [],<br><br>
    '8' : ['9'],<br><br>
    '9' : [],<br><br>
    '10' : []<br><br>
     }<br><br>
visited = []<br><br>
queue = []<br><br>
def bfs(visited, graph, node):<br><br>
    visited.append(node)<br><br>
    queue.append(node)<br><br>
    while queue:<br><br>
        m = queue.pop(0)<br><br>
        print (m, end = " ")<br><br>
        for neighbour in graph[m]:<br><br>
            if neighbour not in visited:<br><br>
                visited.append(neighbour)<br><br>
                queue.append(neighbour)<br><br>
print("Following is the Breadth-First Search")<br><br>
bfs(visited, graph, '1')<br><br>
**Output**<br><br>
Following is the Breadth-First Search<br><br>
1 2 10 3 8 4 9 5 6 7 <br><br>


**TicTacToe**<br><br>
import numpy as np<br><br>
import random<br><br>
from time import sleep<br><br>

def create_board():<br><br>
    return(np.array([[0, 0, 0],<br><br>
              [0, 0, 0],<br><br>
            [0, 0, 0]]))<br><br>

def possibilities(board):<br><br>
    l = []<br>
<br>
    for i in range(len(board)):<br>
        for j in range(len(board)):<br>

            if board[i][j] == 0:<br>
                l.append((i, j))<br>
    return(l)<br>

def random_place(board, player):<br>
    selection = possibilities(board)<br>
    current_loc = random.choice(selection)<br>
    board[current_loc] = player<br>
    return(board)<br>
<br>
def row_win(board, player):<br>
    for x in range(len(board)):<br>
        win = True<br>

        for y in range(len(board)):<br>
            if board[x, y] != player:<br>
                win = False<br>
                continue<br>

        if win == True:<br>
            return(win)<br>
    return(win)<br>
<br>
def col_win(board, player):<br>
    for x in range(len(board)):<br>
        win = True<br>

        for y in range(len(board)):<br>
            if board[y][x] != player:<br>
                win = False<br>
                continue<br>

        if win == True:<br>
            return(win)<br><br>
    return(win)<br>

def diag_win(board, player):<br>
    win = True<br>
    y = 0<br>
    for x in range(len(board)):<br>
        if board[x, x] != player:<br>
            win = False<br>
    if win:<br>
        return win<br>
    win = True<br>
    if win:<br>
        for x in range(len(board)):<br>
            y = len(board) - 1 - x<br>
            if board[x, y] != player:<br>
                win = False<br>
    return win<br>

def evaluate(board):<br>
    winner = 0<br>

    for player in [1, 2]:<br>
        if (row_win(board, player) or<br>
            col_win(board,player) or<br>
            diag_win(board,player)):<br>
            winner = player<br>
    if np.all(board != 0) and winner == 0:<br>
            winner = -1<br>
            
    return winner<br>
def play_game():<br>
    board, winner, counter = create_board(), 0, 1<br>
    print(board)<br>
    sleep(2)<br>

    while winner == 0:<br>
        for player in [1, 2]:<br>
            board = random_place(board, player)<br>
            print("Board after " + str(counter) + " move")<br>
            print(board)<br>
            sleep(2)<br>
            counter += 1<br>
            winner = evaluate(board)<br>
            if winner != 0:<br>
                break<br>
    return(winner)<br>
print("Winner is: " + str(play_game()))<br>

**Output**<br>
[[0 0 0]<br>
 [0 0 0]<br>
 [0 0 0]]<br>
Board after 1 move<br>
[[0 1 0]<br>
 [0 0 0]<br>
 [0 0 0]]<br>
Board after 2 move<br>
[[0 1 0]<br>
 [0 0 0]<br>
 [0 2 0]]<br>
Board after 3 move<br>
[[1 1 0]<br>
 [0 0 0]<br>
 [0 2 0]]<br>
Board after 4 move<br>
[[1 1 0]<br>
 [0 0 2]<br>
 [0 2 0]]<br>
Board after 5 move<br>
[[1 1 0]<br>
 [1 0 2]<br>
 [0 2 0]]<br>
Board after 6 move<br>
[[1 1 0]<br>
 [1 0 2]<br>
 [2 2 0]]<br>
Board after 7 move<br>
[[1 1 0]<br>
 [1 0 2]<br>
 [2 2 1]]<br>
Board after 8 move<br>
[[1 1 0]<br>
 [1 2 2]<br>
 [2 2 1]]<br>
Board after 9 move<br>
[[1 1 1]<br>
 [1 2 2]<br>
 [2 2 1]]<br>
Winner is: 1<br>

from collections import defaultdict<br>
jug1, jug2, aim = 4, 3, 2<br>
visited = defaultdict(lambda: False)<br>
def waterJugSolver(amt1, amt2): <br>
    if (amt1 == aim and amt2 == 0) or (amt2 == aim and amt1 == 0):<br>
        print(amt1, amt2)<br>
        return True<br>
    if visited[(amt1, amt2)] == False:<br>
        print(amt1, amt2)<br>
        visited[(amt1, amt2)] = True<br>
        return (waterJugSolver(0, amt2) or<br>
                waterJugSolver(amt1, 0) or<br>
                waterJugSolver(jug1, amt2) or<br>
                waterJugSolver(amt1, jug2) or<br>
                waterJugSolver(amt1 + min(amt2, (jug1-amt1)),<br>
                amt2 - min(amt2, (jug1-amt1))) or<br>
                waterJugSolver(amt1 - min(amt1, (jug2-amt2)),<br>
                amt2 + min(amt1, (jug2-amt2))))<br>
    else:<br>
        return False<br>
print("Steps: ")<br>
waterJugSolver(0, 0)<br>

**Output**<br>
Steps: <br>
0 0<br>
4 0<br>
4 3<br>
0 3<br>
3 0<br>
3 3<br>
4 2<br>
0 2<br>
True<br>

**8 puzzels**<br>
import copy<br>
from heapq import heappush, heappop<br>
n = 3
row = [ 1, 0, -1, 0 ]<br>
col = [ 0, -1, 0, 1 ]<br>

class priorityQueue:<br>
    def __init__(self):<br>
        self.heap = []<br>
    def push(self, k):<br>
         heappush(self.heap, k)<br>
    def pop(self):<br>
        return heappop(self.heap)
    def empty(self):<br>
        if not self.heap:<br>
            return True<br>
        else:<br>
            return False<br>

class node:<br>
        def __init__(self, parent, mat, empty_tile_pos,cost, level):<br>
            self.parent = parent<br>
            self.mat = mat<br>
            self.empty_tile_pos = empty_tile_pos<br>
            self.cost = cost<br>
            self.level = level<br>
            
        def __lt__(self, nxt):<br>
            return self.cost < nx<br>.cost<br>
def calculateCost(mat, final) -> int:<br>
    count = 0<br>
    for i in range(n):<br>
        for j in range(n):<br>
            if ((mat[i][j]) and (mat[i][j] != final[i][j])):
                count += 1<br>
    return count<br>

def newNode(mat, empty_tile_pos, new_empty_tile_pos,level, parent, final) -> node:
    new_mat = copy.deepcopy(mat)<br>
    x1 = empty_tile_pos[0]<br>
    y1 = empty_tile_pos[1]<br>
    x2 = new_empty_tile_pos[0]<br>
    y2 = new_empty_tile_pos[1]<br>
    new_mat[x1][y1], new_mat[x2][y2] = new_mat[x2][y2], new_mat[x1][y1]<br>
    cost = calculateCost(new_mat, final)<br>
    new_node = node(parent, new_mat, new_empty_tile_pos,cost, level)<br>
    return new_node<br>

def printMatrix(mat):<br>
    for i in range(n):<br>
        for j in range(n):<br>
            print("%d " % (mat[i][j]), end = " ")
        print()<br>

def isSafe(x, y):<br>
    return x >= 0 and x < n and y >= 0 and y < n<br>

def printPath(root):<br>
    if root == None:<br>
        return<br>
    printPath(root.parent)<br>
    printMatrix(root.mat)<br>
    print()<br>
    
def solve(initial, empty_tile_pos, final):<br>
    pq = priorityQueue()<br>
    cost = calculateCost(initial, final)<br>
    root = node(None, initial,empty_tile_pos, cost, 0)<br>
    pq.push(root)<br>
    while not pq.empty():<br>
        minimum = pq.pop()<br>
        if minimum.cost == 0:<br>
            printPath(minimum)<br>
            return<br>
        for i in range(n):<br>
            new_tile_pos = [minimum.empty_tile_pos[0] + row[i],minimum.empty_tile_pos[1] + col[i], ]<br>
            if isSafe(new_tile_pos[0], new_tile_pos[1]):<br>
                child=newNode(minimum.mat,minimum.empty_tile_pos,new_tile_pos,minimum.level+1,minimum,final,)<br>
                pq.push(child)
initial = [ [ 1, 2, 3 ],[ 5, 6, 0 ],[ 7, 8, 4 ] ]<br>
final = [ [ 1, 2, 3 ],[ 5, 8, 6 ],[ 0, 7, 4 ] ]<br>
empty_tile_pos = [ 1, 2 ]<br>
solve(initial, empty_tile_pos, final)<br>

**Output**<br>
1  2  3  <br>
5  6  0  <br>
7  8  4  <br>

1  2  3  <br>
5  0  6  <br>
7  8  4  <br>

1  2  3  <br>
5  8  6  <br>
7  0  4  <br>

1  2  3  <br>
5  8  6  <br>
0  7  4  <br>



**9.Write a program to implement the FIND-S Algorithm for finding the most specific hypothesis based on a given set of training data samples. Read the training data from a .CSV file.**<br>
import pandas as pd<br>
import numpy as np<br>
 <br>
#to read the data in the csv file<br>
data = pd.read_csv("Train.csv")<br>
print(data,"")<br>
 <br>
#making an array of all the attributes<br>
d = np.array(data)[:,:-1]<br>
print("\n The attributes are:\n ",d)<br>
 
#segragating the target that has positive and negative examples<br>
target = np.array(data)[:,-1]<br>
print("\n The target is: ",target)<br>
 
#training function to implement find-s algorithm<br>
def train(c,t):<br>
    for i, val in enumerate(t):<br>
        if val == "Yes":<br>
            specific_hypothesis = c[i].copy()<br>
            break<br>
             
    for i, val in enumerate(c):<br>
        if t[i] == "Yes":<br>
            for x in range(len(specific_hypothesis)):<br>
                if val[x] != specific_hypothesis[x]:<br>
                    specific_hypothesis[x] = '?'<br>
                else:<br>
                    pass<br>
                 
    return specific_hypothesis<br>
 
#obtaining the final hypothesis<br>
print("\n The final hypothesis is:",train(d,target))<br>



**10.Write a program to implement the Candidate-Elimination algorithm, For a given set of training data examples stored in a .CSV file.**<br>
import csv<br><br><br><br><br><br><br><br><br><br><br>
with open("Train.csv")as csv_file:<br><br><br><br><br><br><br><br><br><br>
    #csv_file=csv.reader(f)<br><br><br><br><br><br><br><br><br>
    #data=list(csv_file)<br>
    readcsv=csv.reader(csv_file,delimiter=',')<br><br><br><br><br><br><br>
    data=[]<br><br><br><br><br><br>
    for row in readcsv:<br><br><br><br><br>
        data.append(row)<br><br><br><br>
    s=data[1][:-1]<br><br><br>
    g=[['?'for i in range(len(s))]for j in range(len(s))]<br><br>
    for i in data:<br>
        if i[-1]=="Yes":<br>
            for j in range(len(s)):<br>
                if i[j]!=s[j]:<br>
                    s[j]='?'<br>
                    g[j][j]='?'<br>
        elif i[-1]=="No":<br>
            for j in range(len(s)):<br>
                if i[j]!=s[j]:<br>
                      g[j][j]=s[j]<br>
                else:<br>
                    g[j][j]="?"<br>
        print("\n steps of candidate elimination algorithm",data.index(i)+1)<br>
        print(s)<br>
        print(g)<br>
    gh=[]<br>
    for i in g:<br>
        for j in i:<br>
        
            if j!='?':<br>
            
                gh.append(i)<br>
                
                break<br>
                
    print("\nFinal specific hypothesis:\n",s)<br>
    
    print("\nFinal general hypothesis:\n",gh)   <br>
    
    steps of candidate elimination algorithm 1<br>
['Sunny', 'Warm', '?', 'Strong', 'Warm', 'Same']<br>

[['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?']]<br>


 steps of candidate elimination algorithm 2<br>
 
['Sunny', 'Warm', '?', 'Strong', 'Warm', 'Same']<br>

[['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?']]<br>


 steps of candidate elimination algorithm 3<br>
 
['Sunny', 'Warm', '?', 'Strong', 'Warm', 'Same']<br>

[['Sunny', '?', '?', '?', '?', '?'], ['?', 'Warm', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', 'Same']]<br>


 steps of candidate elimination algorithm 4<br>
 
['Sunny', 'Warm', '?', 'Strong', '?', '?']<br>

[['Sunny', '?', '?', '?', '?', '?'], ['?', 'Warm', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?']]<br>


Final specific hypothesis:<br>

 ['Sunny', 'Warm', '?', 'Strong', '?', '?']<br>
 

Final general hypothesis:<br>

 [['Sunny', '?', '?', '?', '?', '?'], ['?', 'Warm', '?', '?', '?', '?']]<br>
 <br>
 
 
 
 
 8 puzzles<br>
# Python3 program to print the path from root<br>
# node to destination node for N*N-1 puzzle<br>


# algorithm using Branch and Bound
<br>
# The solution assumes that instance of
# puzzle is solvable<br>

# Importing copy for deepcopy function<br>
import copy<br>

# Importing the heap functions from pytho<br>
# library for Priority Queue<br>
from heapq import heappush, heappop<br>

# This variable can be changed to change<br><br>
# the program from 8 puzzle(n=3) to 15<br>
# puzzle(n=4) to 24 puzzle(n=5)...<br>
n = 3<br>

# bottom, left, top, right<br>
row = [ 1, 0, -1, 0 ]<br>
col = [ 0, -1, 0, 1 ]<br>

# A class for Priority Queue<br>
class priorityQueue:<br>
	
	# Constructor to initialize a<br>
	# Priority Queue<br>
	def __init__(self):<br>
		self.heap = []<br>

	# Inserts a new key 'k'<br>
	def push(self, k):<br>
		heappush(self.heap, k)<br>

	# Method to remove minimum element<br>
	# from Priority Queue<br>
	def pop(self):<br>
		return heappop(self.heap)<br>

	# Method to know if the Queue is empty<br>
	def empty(self):<br>
		if not self.heap:<br>
			return True<br>
		else:<br>
			return False<br>

# Node structure<br>
class node:<br>
	
	def __init__(self, parent, mat, empty_tile_pos,<br>
				cost, level):<br>
					
		# Stores the parent node of the<br>
		# current node helps in tracing<br>
		# path when the answer is found<br>
		self.parent = parent<br>

		# Stores the matrix<br>
		self.mat = mat<br>

		# Stores the position at which the<br>
		# empty space tile exists in the matrix<br>
		self.empty_tile_pos = empty_tile_pos<br>

		# Stores the number of misplaced tiles<br>
		self.cost = cost<br>

		# Stores the number of moves so far<br>
		self.level = level<br>

	# This method is defined so that the<br>
	# priority queue is formed based on<br>
	# the cost variable of the objects<br>
	def __lt__(self, nxt):<br>
		return self.cost < nxt.cost<br>

# Function to calculate the number of<br>
# misplaced tiles ie. number of non-blank<br>
# tiles not in their goal position<br>
def calculateCost(mat, final) -> int:<br>
	
	count = 0<br>
	for i in range(n):<br>
		for j in range(n):<br>
			if ((mat[i][j]) and<br>
				(mat[i][j] != final[i][j])):<br>
				count += 1<br>
				
	return count<br>

def newNode(mat, empty_tile_pos, new_empty_tile_pos,<br>
			level, parent, final) -> node:<br>
				
	# Copy data from parent matrix to current matrix<br>
	new_mat = copy.deepcopy(mat)<br>

	# Move tile by 1 position<br>
	x1 = empty_tile_pos[0]<br>
	y1 = empty_tile_pos[1]<br>
	x2 = new_empty_tile_pos[0]<br>
	y2 = new_empty_tile_pos[1]<br>
	new_mat[x1][y1], new_mat[x2][y2] = new_mat[x2][y2], new_mat[x1][y1]<br>

	# Set number of misplaced tiles<br>
	cost = calculateCost(new_mat, final)<br>

	new_node = node(parent, new_mat, new_empty_tile_pos,<br>
					cost, level)<br>
	return new_node<br>

# Function to print the N x N matrix<br>
def printMatrix(mat):<br>
	
	for i in range(n):<br>
		for j in range(n):<br>
			print("%d " % (mat[i][j]), end = " ")<br>
			
		print()<br>

# Function to check if (x, y) is a valid<br>
# matrix coordinate<br>
def isSafe(x, y):<br>
	
	return x >= 0 and x < n and y >= 0 and y < n<br>

# Print path from root node to destination node<br>
def printPath(root):<br>
	
	if root == None:<br>
		return<br>
	
	printPath(root.parent)<br>
	printMatrix(root.mat)<br>
	print()<br>

# Function to solve N*N - 1 puzzle algorithm<br>
# using Branch and Bound. empty_tile_pos is<br>
# the blank tile position in the initial state.<br>
def solve(initial, empty_tile_pos, final):<br>
	
	# Create a priority queue to store live<br>
	# nodes of search tree<br>
	pq = priorityQueue()<br>

	# Create the root node<br>
	cost = calculateCost(initial, final)<br>
	root = node(None, initial,<br>
				empty_tile_pos, cost, 0)<br>

	# Add root to list of live nodes<br>
	pq.push(root)<br>

	# Finds a live node with least cost,<br>
	# add its children to list of live<br>
	# nodes and finally deletes it from<br>
	# the list.<br>
	while not pq.empty():<br>

		# Find a live node with least estimated<br>
		# cost and delete it form the list of<br>
		# live nodes<br>
		minimum = pq.pop()<br>

		# If minimum is the answer node<br>
		if minimum.cost == 0:<br>
			
			# Print the path from root to<br><br>
			# destination;<br>
			printPath(minimum)<br>
			return<br>

		# Generate all possible children<br>
		for i in range(4):<br>
			new_tile_pos = [<br>
				minimum.empty_tile_pos[0] + row[i],<br>
				minimum.empty_tile_pos[1] + col[i], ]<br>
				
			if isSafe(new_tile_pos[0], new_tile_pos[1]):<br>
				
				# Create a child node<br>
				child = newNode(minimum.mat,<br>
								minimum.empty_tile_pos,<br>
								new_tile_pos,<br>
								minimum.level + 1<br>
								minimum, final,)<br>

				# Add child to list of live nodes<br>
				pq.push(child)<br>

# Driver Code<br>

# Initial configuration<br>
# Value 0 is used for empty space<br>
initial = [ [ 1, 2, 3 ],<br>
			[ 5, 6, 0 ],<br>
			[ 7, 8, 4 ] ]<br>

# Solvable Final configuration<br>
# Value 0 is used for empty space<br>
final = [ [ 1, 2, 3 ],<br>
		[ 5, 8, 6 ],<br>
		[ 0, 7, 4 ] ]<br>

# Blank tile coordinates in<br>
# initial configuration<br>
empty_tile_pos = [ 1, 2 ]<br>

# Function call to solve the puzzle<br>
solve(initial, empty_tile_pos, final)<br>

# This code is contributed by Kevin Joshi<br>
1  2  3  <br>
5  6  0  <br>
7  8  4  <br>

1  2  3  <br>
5  0  6  <br>
7  8  4  <br>

1  2  3  <br>
5  8  6  <br>
7  0  4  <br>

1  2  3 <br> 
5  8  6  <br>
0  7  4  <br>

 
 
from sys import maxsize
from itertools import permutations
V=4

def TravellingSalesmanProblem(graph, s):
    vertex=[]
    for i in range(V):
        if i!=s:
            vertex.append(i)
            
    min_path=maxsize
    next_permutation=permutations(vertex)
    for i in next_permutation:
        current_pathweight=0
        k=s
        for j in i:
            current_pathweight+=graph[k][j]
            k=j
        current_pathweight+=graph[k][s]   
        min_path=min(min_path, current_pathweight)
        
    return min_path

if __name__ == "__main__":
    graph = [[0, 10, 15, 20], [10, 0, 35, 25],
             [15, 35, 0, 30], [20, 25, 30, 0]]
    s=0
    print(TravellingSalesmanProblem(graph, s))
    
