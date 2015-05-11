# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        # self.values can be used to get the maximum value

        "*** YOUR CODE HERE ***"

        # V(k+1)(s) = max( sum(T(s,a,s') *(R(s,a,s') + gamma*Vk(s'))) forall s')
        # for all a

        allStates = mdp.getStates()

        v = [[0 for j in range(iterations+1)] for i in range(len(allStates))]

        mapz = {}
        i = 0
        for state in allStates:
            mapz[state] = i
            i+=1
            
        for i in range(1,self.iterations+1):
            for state in allStates:
                
                self.values[state] = 0

                maxz = util.Counter() #action -> Q
                
                for action in mdp.getPossibleActions(state):
                    
                    # t is a list of (nextState, prob) tuples
                    t = mdp.getTransitionStatesAndProbs(state, action)

                    tot = util.Counter() #nexstate->sum
                    
                    for nex in t:
                        r = mdp.getReward(state,action,nex[0])
                        if nex[0] not in self.values: self.values[nex[0]] = 0
                        tot[nex[0]] = nex[1] * (r + self.discount * v[mapz[nex[0]]][i-1])
                            
                    maxz[action] = tot.totalCount()

                v[mapz[state]][i] = maxz[maxz.argMax()]
        for state in allStates:
            self.values[state] = v[mapz[state]][iterations]

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        # Qstar(s) = sum(Tstar(s,a,s') * (R(s,a,s') + gamma*V*(s')) ) for all s'
        tot = 0
        t = self.mdp.getTransitionStatesAndProbs(state, action)
        for nex in t:
            r = self.mdp.getReward(state,action,nex[0]) 
            tot += nex[1] * (r + self.discount * self.values[nex[0]])

        return tot
    
    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        dic = util.Counter()
        for action in self.mdp.getPossibleActions(state):
            dic[action] = self.computeQValueFromValues(state,action)

        return dic.argMax()
        
    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
