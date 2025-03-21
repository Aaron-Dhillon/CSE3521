## Authors: Ben Varkey, Aaron Dhillon

import time
import heapq as hq
import pandas as pd
import numpy as np
from typing import Union
import copy as cp

class Node:
        def __init__(self, state, parent = None, cost = 0, h = 0):
            self.state = state #Country Name | Current Board
            self.parent = parent #Parent Country Name | Previous Board
            self.actions = None #Adjacent Countrys | Possible moves
            self.cost = cost #Cost to travel from parent | Number of moves
            self.h = h #SLD | Manhattan Distance


        def __lt__(self, other):
            if(isinstance(other, Node)):
                return self.cost < other.cost
            return False

class Travel:  
    def __init__(self,init_state, goal_state):
        self.data = pd.read_csv("MapInfo.csv", header=0, index_col=0)
        self.init_node = Node(state =init_state, h = self.getHeuristic(init_state)) #set when parsing csv
        self.goal_state = goal_state
    

    def getHeuristic(self, state):
        row = self.data.loc[state]
        return row["SLD"]

    def generateActions(self,node:Node):
        actions = []
        c_name = node.state
        c_row = self.data.loc[c_name]
        #go through the the CSV and get the adjacent countries and respectiev cost
        for column in self.data.columns:
            entry = c_row[column]
            #if we are on an entry thats non empty which isnt the SLD or label column
            if pd.notna(entry) and column != c_name and column != "SLD":
                action = Node(state=column,parent=node,cost=entry+node.cost, h = self.getHeuristic(column) )
                actions.append(action)
        node.actions = actions 

        
    def goal_test(self,node:Node):
        return node.state== self.goal_state
    
    #prints solution
    def solution(self,node:Node,numExpNodes):
        n = node
        output = n.state
        #moves up the path printing apopending each parent
        while(n.parent is not None):
            output = n.parent.state + " to " + output
            n = n.parent
        output = output + ":\t" + str(numExpNodes) + " expanded (including goal state)"
        print(output)
    
class Puzzle:
    #input should already be a list
    def __init__(self,input,n):
        
        #splits input into n rows
        init_state= self.make(input,n)
        self.init_node = Node(state = init_state, h = self.getHeuristic(init_state))
        
        self.goal_state =[] 
        #makes the goal state
        for i in range(0,n*n):
            self.goal_state.append(str(i))
        self.goal_state = self.make(self.goal_state,n)

    #used to make the initial state
    def make(self,input,n):
        init_state = []
        row =[]
        
        for i in range(1, len(input)+1):
            row.append(input[i-1])
            #split into n rows
            if i % n == 0  and i !=0:
                init_state.append(row)
                row = []
            
        return init_state
    
    #gets Manhattan Distance Hueristic
    def getHeuristic(self, state):
        n = len(state)
        sum = 0
        for r in range(0, n):
            row = state[r]
            for c in range(0, n):
                entry = int(row[c])
                target_r = entry//n
                target_c = entry%n
                #Adds difference between column indexes and row indexes 
                sum = sum + (np.abs(target_r - r)+np.abs(target_c - c))
    
        return sum
    
    #This Hueristic counts the number of spaces away the entry is from its proper entry, but only counting horozontally
    #You start counting on the tile next to the current tile going across a row. Then once you reach the end
    #of a row, you continue counting across the next row staring at the first column and so on until you reach goal entry.
    #So if entry 8 is in slot [0,0], its hueritic would be 8 since you coul start at [0,0] and couldnt across each row until
    #destination is reached
    def otherHueristic(self, state):
        n = len(state)
        sum = 0 
        index = 0
        for r in range(0,n):
            for c in range(0,n):
                entry = int(state[r][c])
                sum = sum + (abs(entry - index))
                index = index+1
        return sum 
    

    def generateActions(self, node:Node):        
        #find 0 index and row
        #generate states where the row/s next to 0 row have their values swapped by 0
        actions = []
        curr_state = node.state
        n = len(curr_state)
        for r in range(0, n):
            for c in range(0, n):
                if curr_state[r][c] == '0':
                    #possible directions
                    x_dir = [0,0,-1,1]
                    y_dir = [-1,1,0,0]
                    for move in range(0,4):
                        new_r = r + x_dir[move]
                        new_c = c + y_dir[move]
                        #cheakc if move is valid (entry is still in bounds) and adds state if it is
                        if new_r < n and new_r > -1 and new_c < n and new_c > -1:
                            new_state= cp.deepcopy(curr_state)
                            new_state[r][c] = curr_state[new_r][new_c]
                            new_state[new_r][new_c] = '0'
                            new_node = Node(state = new_state, cost = node.cost +1, parent = node, h = self.getHeuristic(new_state))
                            actions.append(new_node)

        node.actions= actions


    def goal_test(self,node:Node):
        return node.state == self.goal_state
    
    def solution(self,node:Node,numExpNodes):
        #returns a board as a string
        def print_state(state):
            output = ""
            for row in state:
                output = output + ("| " + " | ".join(row) + " | \n")
            output = output + "\n"
            return output


        n = node
        output = print_state(n.state)
        while(n.parent is not None):
            output = print_state(n.parent.state) + "    to \n\n"  + output
            n = n.parent

        output = output + str(numExpNodes) + " expanded (including goal state)"
        print(output)
    

class Search:

        @staticmethod
        def search(problem:Union[Travel,Puzzle], is_ucs:bool):
            node = problem.init_node
            frontier = []
            #inserting initial node into queue
            if is_ucs:
                hq.heappush(frontier,(node.cost, node))
            else:
                hq.heappush(frontier,(node.cost + node.h, node))
            explored = {}
            while(1):
                
                #fail check
                if(len(frontier)==0):
                     print("Search Failed")
                     return 1
                
                node = (hq.heappop(frontier))[1]
                #test if we are at goal state
                if(problem.goal_test(node)):
                    explored[str(node.state)]  = node.cost
                    problem.solution(node, len(explored))
                    return 1
                #explore node and get action
                explored[str(node.state)] = node.cost
                problem.generateActions(node)

                for child in node.actions:
                    state = str(child.state)
                    # check if action node is in explored
                    if state not in explored:
                        if is_ucs:
                            hq.heappush(frontier, (child.cost, child))
                        else:
                            hq.heappush(frontier, (child.cost +child.h, child))
                    elif child.cost < explored[state]:
                         #update cost in explored
                         explored[state] = child.cost
                         size = len(frontier)
                         #go through frintier and see if their is a node with same 
                         #state and higher cost then child node and if there is 
                         #replace it
                         for i in range(1,size):
                            n = (hq.heappop(frontier))[1]

                            
                            if is_ucs:
                                if n.state == child.state:
                                    hq.heappush(frontier,(child.cost, child))
                                    
                                else:
                                    hq.heappush(frontier,(n.cost, n))
                            elif not is_ucs:
                                if n.state == child.state:
                                    hq.heappush(frontier,(child.cost + child.h,child))
                                else:
                                     hq.heappush(frontier,(n.cost + n.h,n))
                                   
                              
def main():
    problem_type = input("Pick a Problem. Enter \"Traveling\" or \"Puzzle\": ")

    while problem_type != "Traveling" and problem_type != "Puzzle":
        problem_type = input("Enter \"Traveling\" or \"Puzzle\": ")

    if problem_type == "Traveling":
        init_state = input("Starting Country: ")
        goal_state = input("Destination: ")
        search_method= input("UCS or A* search: ")
        problem = Travel(init_state,goal_state)
        if search_method == "UCS":
            Search.search(problem,True)
        elif search_method == "A*":
            Search.search(problem, False)
    elif problem_type == "Puzzle":
        n = input("Input N: ")
        print("Input intial state as a series of numbers seperated by spaces. Use 0 for the empty space.")
        print("Numbers will be input from left to right until a row is filled; then the next row will be filled")
        print("So in a 3x3 Puzzle, the 6th number (starting from 0) will correspond to the 1st element in the 3rd row.")
        inp = input("Input intial state: ")
        search_method= input("UCS or A* search: ")
        init_state = inp.split()
        problem = Puzzle(init_state,int(n))
        if search_method == "UCS":
            start = time.time()
            Search.search(problem, True)
            end = time.time()
            print(f"Time: {end - start} seconds")
        elif search_method == "A*":
            Search.search(problem, False)


main()
    