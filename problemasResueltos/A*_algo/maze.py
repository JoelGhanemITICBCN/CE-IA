import pyamaze as maze
import time
from queue import PriorityQueue 

ROWS = 20
COLS = 20

#Defining the distance between cells in order to Know the remaining distance
def distance(cell1, cell2):
     return abs(cell1[0] - cell2[0]) + abs(cell1[1] - cell2[1])
    

def aStar(m,n_rows,n_cols):
    start = (n_rows,n_cols)
    end = (1,1)
    pq = PriorityQueue()
    pq.put((0,start))

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
