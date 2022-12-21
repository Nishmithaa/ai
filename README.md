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
