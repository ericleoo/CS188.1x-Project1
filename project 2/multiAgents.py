# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action) #object pacman.Gamestate
        newPos = successorGameState.getPacmanPosition() #tuple(x,y)
        newFood = successorGameState.getFood() #object game.Grid #list of tuples.
        newGhostStates = successorGameState.getGhostStates() #A list of game.AgentState
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates] #list
        

        oldFood = currentGameState.getFood()
        if action == 'Stop':
            return -float("inf")

        for ghostState in newGhostStates:
            if ghostState.getPosition() == newPos and ghostState.scaredTimer is 0:
                return -float("inf")

        aList = []
        for food in oldFood.asList():
            aList.append(util.manhattanDistance(food,newPos))
        return 1.0/float(min(aList)+1)
        
def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """
    pacmanIndex = 0
    
    
    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game

          gameState.isWin():
            Returns whether or not the game state is a winning state

          gameState.isLose():
            Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        # Collect legal moves and successor states

        
        curDepth = 0
        currentAgentIndex = 0
        val = self.value(gameState,currentAgentIndex,curDepth)
        return val[0]
    
    def value(self,gameState,currentAgentIndex,curDepth):
        if currentAgentIndex >= gameState.getNumAgents():
            currentAgentIndex = 0
            curDepth += 1
        if curDepth == self.depth:
            # Terminal Node
            return self.evaluationFunction(gameState)
        if currentAgentIndex == self.pacmanIndex:
            # If agent is pacman, maximize
            return self.maxValue(gameState, currentAgentIndex, curDepth)
        else: # else minimize
            return self.minValue(gameState,currentAgentIndex, curDepth)
    def maxValue(self,gameState,currentAgentIndex,curDepth):
        v = ("unknown", -1*float("inf"))

        if not gameState.getLegalActions(currentAgentIndex):
            # if there are no legal actions -> terminal node.
            return self.evaluationFunction(gameState)
        
        for action in gameState.getLegalActions(currentAgentIndex):
            if action == "Stop": continue
            
            retVal = self.value(gameState.generateSuccessor(currentAgentIndex,
                        action),currentAgentIndex + 1, curDepth)
            if type(retVal) is tuple:
                retVal = retVal[1]
            vNew = max(v[1],retVal)

            if vNew != v[1]:
                v = (action,vNew)
        return v
            
    def minValue(self,gameState,currentAgentIndex,curDepth):
        v = ("unknown", 1*float("inf"))

        if not gameState.getLegalActions(currentAgentIndex):
            return self.evaluationFunction(gameState)
        for action in gameState.getLegalActions(currentAgentIndex):
            if action == "Stop": continue
            retVal = self.value(gameState.generateSuccessor(currentAgentIndex,
                                                            action),
                                currentAgentIndex + 1,
                                curDepth)
            
            if type(retVal) is tuple:
                retVal = retVal[1]

            #print "ACTION:",action,"VAL:",retVal
            
            vNew = min(v[1] , retVal)
            if vNew != v[1]:
                v = (action,vNew)
        return v
        
class AlphaBetaAgent(MultiAgentSearchAgent):
    
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    pacmanIndex = 0
    
    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        curDepth = 0
        currentAgentIndex = 0
        minINF = -1*float("inf")
        maxINF = float("inf")
        val = self.value(gameState,currentAgentIndex,curDepth, minINF, maxINF)
        return val[0]
    
    def value(self,gameState,currentAgentIndex,curDepth,alpha,beta):
        if currentAgentIndex >= gameState.getNumAgents():
            currentAgentIndex = 0
            curDepth += 1
        if curDepth == self.depth:
            # Terminal Node
            return self.evaluationFunction(gameState)
        if currentAgentIndex == self.pacmanIndex:
            # If agent is pacman, maximize
            return self.maxValue(gameState, currentAgentIndex, curDepth,alpha,beta)
        else: # else minimize
            return self.minValue(gameState,currentAgentIndex, curDepth,alpha,beta)
    def maxValue(self,gameState,currentAgentIndex,curDepth,alpha,beta):
        v = ("empty", -1*float("inf"))

        if not gameState.getLegalActions(currentAgentIndex):
            # if there are no legal actions -> terminal node.
            return self.evaluationFunction(gameState)
        
        for action in gameState.getLegalActions(currentAgentIndex):
            if action == "Stop": continue

            if beta < alpha:
                break
            
            retVal = self.value(gameState.generateSuccessor(currentAgentIndex,
                        action),currentAgentIndex + 1, curDepth,alpha,beta)
            if type(retVal) is tuple:
                retVal = retVal[1]
            vNew = max(v[1],retVal)
            
            if vNew != v[1]:
                v = (action,vNew)

            if vNew > beta: return v
            alpha = max(alpha, vNew)
            #alpha = max(alpha,vNew)
            #if beta < alpha:
            #    break

        return v
            
    def minValue(self,gameState,currentAgentIndex,curDepth,alpha,beta):
        v = ("empty", 1*float("inf"))

        if not gameState.getLegalActions(currentAgentIndex):
            return self.evaluationFunction(gameState)
        for action in gameState.getLegalActions(currentAgentIndex):
            if action == "Stop": continue

            if beta < alpha:
                break
                
            retVal = self.value(gameState.generateSuccessor(currentAgentIndex,
                                                            action),
                                currentAgentIndex + 1,
                                curDepth,alpha,beta)
            if type(retVal) is tuple:
                retVal = retVal[1]
            vNew = min(v[1] , retVal)
            if vNew != v[1]:
                v = (action,vNew)
            if vNew < alpha: return v
            beta = min(beta, vNew)
            #beta = min(beta,vNew)
            #if beta < alpha:
            #    break
        return v
        
class ExpectimaxAgent(MultiAgentSearchAgent):

    pacmanIndex = 0
    
    
    def getAction(self, gameState):
        # Collect legal moves and successor states
        curDepth = 0
        currentAgentIndex = 0
        val = self.value(gameState,currentAgentIndex,curDepth)
        return val[0]
    
    def value(self,gameState,currentAgentIndex,curDepth):
        if currentAgentIndex >= gameState.getNumAgents():
            currentAgentIndex = 0
            curDepth += 1
        if curDepth == self.depth:
            # Terminal Node
            return self.evaluationFunction(gameState)
        if currentAgentIndex == self.pacmanIndex:
            # If agent is pacman, maximize
            return self.maxValue(gameState, currentAgentIndex, curDepth)
        else: # else minimize
            return self.expValue(gameState,currentAgentIndex, curDepth)
    def maxValue(self,gameState,currentAgentIndex,curDepth):
        v = ("unknown", -1*float("inf"))

        if not gameState.getLegalActions(currentAgentIndex):
            # if there are no legal actions -> terminal node.
            return self.evaluationFunction(gameState)

        
        for action in gameState.getLegalActions(currentAgentIndex):
            if action == "Stop": continue
            
            retVal = self.value(gameState.generateSuccessor(currentAgentIndex,
                        action),currentAgentIndex + 1, curDepth)
            if type(retVal) is tuple:
                retVal = retVal[1]
            #print "ACTION:",action,"VAL:",retVal
            
            vNew = max(v[1],retVal)
            if vNew != v[1]:
                v = (action,vNew)
        return v
            
    def expValue(self,gameState,currentAgentIndex,curDepth):
        v = ("unknown", 0)

        if not gameState.getLegalActions(currentAgentIndex):
            return self.evaluationFunction(gameState)
        tot = 0
        aList = gameState.getLegalActions(currentAgentIndex)
        bList = []
        for action in aList:
            if action == "Stop": continue
            retVal = self.value(gameState.generateSuccessor(currentAgentIndex,
                                                            action),
                                currentAgentIndex + 1,
                                curDepth)
            if type(retVal) is tuple:
                retVal = retVal[1]
            tot += retVal
            bList.append((action,retVal))
            
        avg = float(tot)/float(len(aList))
        minim = float("inf")
        v = ("unknown", avg)
        for tup in bList:
            if abs(tup[1] - avg) < minim:
                v = (tup[0],avg)
                minim = abs(tup[1]-avg)
        
        return v

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    
    curPos = currentGameState.getPacmanPosition()
    curGhostStates = currentGameState.getGhostStates() #A list of game.AgentState
    curScaredTimes = [ghostState.scaredTimer for ghostState in curGhostStates] #list
    legalActions = currentGameState.getLegalActions()
    curFood = currentGameState.getFood()
    curCaps = currentGameState.getCapsules()
    numOfScaredGhosts = 0

    distanceToFood = []
    distanceToNearestGhost = []

    for ghostState in curGhostStates:
        if ghostState.scaredTimer == 0:
            numOfScaredGhosts += 1
            distanceToNearestGhost.append(0)
            continue
        if util.manhattanDistance(curPos,ghostState.getPosition()) == 0:
            distanceToNearestGhost.append(0)
        else:
            distanceToNearestGhost.append(-1.0/util.manhattanDistance(curPos,ghostState.getPosition()))
    
    for food in curFood.asList():
        distanceToFood.append(-1*util.manhattanDistance(food,curPos))

    if not distanceToFood:
        distanceToFood.append(0)
        
    return max(distanceToFood) + min(distanceToNearestGhost) + currentGameState.getScore() - 100*len(curCaps) - 20*(len(curGhostStates) - numOfScaredGhosts)
    
    
# Abbreviation
better = betterEvaluationFunction

