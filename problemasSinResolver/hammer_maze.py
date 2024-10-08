
"""
In this case the agent will have a hammer making him able to break a single wall wherever he wants
the trick will be that using it will cost him 5 moves
"""
import pyamaze as maze
import time
from queue import PriorityQueue 

ROWS = 20
COLS = 20
#Using the removeWallInBetween function 
def removeWallinBetween(cell1,cell2):
    if cell1 not in m.maze_map or cell2 not in m.maze_map:
        return
    '''
    To remove wall in between two cells
    '''
    if cell1[0]==cell2[0]:
        if cell1[1]==cell2[1]+1:
            m.maze_map[cell1]['W']=1
            m.maze_map[cell2]['E']=1
        else:
            m.maze_map[cell1]['E']=1
            m.maze_map[cell2]['W']=1
    else:
        if cell1[0]==cell2[0]+1:
            m.maze_map[cell1]['N']=1
            m.maze_map[cell2]['S']=1
        else:
            m.maze_map[cell1]['S']=1
            m.maze_map[cell2]['N']=1
#Defining the distance between cells in order to Know the remaining distance
def distance(cell1, cell2):
     return abs(cell1[0] - cell2[0]) + abs(cell1[1] - cell2[1])
    

def aStar(m,n_rows,n_cols):
    start = (n_rows,n_cols)
    end = (1,1)
    pq = PriorityQueue()
    pq.put((0,start))
    hammer_uses = 0

    #Dictionary where I define every cell distance at infite so I can always have the shortest path
    walked = {cell: float('inf') for cell in m.grid}
    walked[start] = 0
    
    #Defining the remaining cells until reaching the end
    remaining = {cell: float('inf') for cell in m.grid}
    remaining[start] = distance(start,end)

    #Auxiliar dictionary to check the shortest path to the end point
    came_from = {}

    directions = {
        'E':(0,1),
        'N':(-1,0),
        'S':(1,0),
        'W':(0,-1)
        }

    while not pq.empty():
        current = pq.get()[1]

        #In case I reach the end I "walk" the path backwards
        if current == end:
            forwardPath = {}

            #While there's a previous point I set the actual point as the previous point
            while current in came_from:
                forwardPath[came_from[current]] = current

                #I set the current point as the past one
                current = came_from[current]
            return forwardPath

        #I check for walls in the maze on every direction next to the point
        for direction, (dr,dc) in directions.items():

            #If its clear I create a neighbor 
            if m.maze_map[current][direction] == 1:
                neighbor = (current[0] + dr, current[1] + dc)

                #I add the distance to the next step to the list
                tentative_walk = walked[current] + 1

                """
                In case the possible way is better than the current, I add the current point to my trail
                I update the distance I have travelled and recalculate the remaining distance
                """

                if tentative_walk < walked[neighbor] :
                    came_from[neighbor] = current
                    walked[neighbor] = tentative_walk
                    remaining[neighbor] = tentative_walk + distance(neighbor,end)
                    pq.put((remaining[neighbor],neighbor))
            else:
                #if hammer_uses == 0:
                neighbor = (current[0] + dr, current[1] + dc) #I add the distance to the next step to the list
                if neighbor in m.maze_map:
                    
                    print("I am breaking the wall between ",current ,neighbor)
                    removeWallinBetween(current,neighbor)
                    hammer_uses += 1
                    tentative_walk = walked[current] + 5

                    if tentative_walk < walked[neighbor] :
                        came_from[neighbor] = current
                        walked[neighbor] = tentative_walk
                        remaining[neighbor] = tentative_walk + distance(neighbor,end)
                        pq.put((walked[neighbor],neighbor))


    return {}
m=maze.maze(ROWS,COLS)
m.CreateMaze()
pre_Astar = time.time()
path = aStar(m,ROWS,COLS)
post_Astar = time.time()
print("A* algorythm execution time is:",post_Astar - pre_Astar)
a=maze.agent(m,footprints=True)
m.tracePath({a:path},delay=5)
m.run()
