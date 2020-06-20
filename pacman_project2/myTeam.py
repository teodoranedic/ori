# myTeam.py
# ---------
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


from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions, Actions
import game
from util import nearestPoint


#################
# Team creation #
#################


def createTeam(firstIndex, secondIndex, isRed,
               first='OffensiveAgent', second='DefensiveAgent'):
    """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

    # The following line is an example only; feel free to change it.
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


##########
# Agents #
##########


class BaseAgent(CaptureAgent):
    """
  A base class for reflex agents that chooses score-maximizing actions
  """

    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)

    def chooseAction(self, gameState):
        """
    Picks among the actions with the highest Q(s,a).
    """
        actions = gameState.getLegalActions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(gameState, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        foodLeft = len(self.getFood(gameState).asList())

        if foodLeft <= 2:
            bestDist = 9999
            for action in actions:
                successor = self.getSuccessor(gameState, action)
                pos2 = successor.getAgentPosition(self.index)
                dist = self.getMazeDistance(self.start, pos2)
                if dist < bestDist:
                    bestAction = action
                    bestDist = dist
            return bestAction

        return random.choice(bestActions)

    def getSuccessor(self, gameState, action):
        """
    Finds the next successor which is a grid position (location tuple).
    """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def evaluate(self, gameState, action):
        """
    Computes a linear combination of features and feature weights
    """
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features * weights

    def getFeatures(self, gameState, action):
        """
    Returns a counter of features for the state
    """
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        features['successorScore'] = self.getScore(successor)
        return features

    def getWeights(self, gameState, action):
        """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
        return {'successorScore': 1.0}


class OffensiveAgent(BaseAgent):

    def getFeatures(self, gameState, action):
        # Start like getFeatures of OffensiveReflexAgent
        features = util.Counter()

        # Get other variables for later use
        food = self.getFood(gameState)
        capsules = gameState.getCapsules()
        foodList = food.asList()
        walls = gameState.getWalls()
        currentPosition = gameState.getAgentState(self.index).getPosition()
        vector = Actions.directionToVector(action)
        newPosition = (int(currentPosition[0] + vector[0]), int(currentPosition[1] + vector[1]))

        # Get set of invaders and defenders
        enemies = [gameState.getAgentState(a) for a in self.getOpponents(gameState)]
        #napdaci
        ghosts = [a for a in enemies if not a.isPacman and a.getPosition() is not None]
        #branitelji
        pacmans = [a for a in enemies if a.isPacman and a.getPosition() is not None]

        # Check if pacman has stopped
        if action == Directions.STOP:
            features["stop"] = 1.0
        successor = self.getSuccessor(gameState, action)

        # Get ghosts close by

        #if u r pacman and u have scared ghosts run baby to them
        # MI SMO PAACMAN
        # 1 i sad gledamo da li imamo uplasene duhove i da li je > 0, > 2 komsije, to da ne bi stigli do njega a on postao normalan,
        #
        #2 sa normalnim duhovima

        for i in ghosts:

            if (i.scaredTimer > 0) and (newPosition == i.getPosition()): # jede duha na tacnoj poziciji
                features['eatGhost'] += 5
                features['eatFood'] = 0
                features['successorPosition'] = -9

            elif (i.scaredTimer > 2) and (newPosition in Actions.getLegalNeighbors(i.getPosition(), walls)):  # jede duha u blizini
                features["scaredGhosts"] += 1
                features["normalGhosts"] = 5    #proba
                features['successorPosition'] = -9

            if (i.scaredTimer == 0) and (newPosition == i.getPosition()):  # nemamo uplasenih duhova
                features["scaredGhosts"] = 0
                features["normalGhosts"] += 5   #proba
                features['successorPosition'] = -9
                if (len(successor.getLegalActions(self.index)) < 3):
                    features['successorPosition'] += 50
            elif (i.scaredTimer == 0) and (newPosition in Actions.getLegalNeighbors(i.getPosition(), walls)):  # neprijatelj u blizini
                #features["scaredGhosts"] += 1
                features["normalGhosts"] += 5  # proba
                if(len(successor.getLegalActions(self.index)) < 4): # kako ga neprijatelj ne bi oterao u cosak
                    print( str(len(successor.getLegalActions(self.index))))
                    features['successorPosition'] += 50

        # ako smo pakman i ako je duh uplasen
        # ako smo pakman i ako nije duh uplasen

        notScared = gameState.getAgentState(self.index).scaredTimer == 0
        # enemies ukoliko smo plavi to su svi crveni bilo duh bilo pakman
        for i in enemies:
            if i.getPosition() is not None:
                if newPosition in Actions.getLegalNeighbors(i.getPosition(), walls) and not notScared:
                    features["closeInvader"] += -10
                    features["eatInvader"] = -10
                elif newPosition == i.getPosition() and not notScared:
                    features["eatInvader"] = -10

        for i in pacmans:
            if newPosition == i.getPosition() and notScared:
                features["eatInvader"] = 1
            elif newPosition in Actions.getLegalNeighbors(i.getPosition(), walls) and notScared:
                features["closeInvader"] += 1


        # Get capsules when nearby
        for c in capsules:
            if newPosition == c and successor.getAgentState(self.index).isPacman:
                features["eatCapsule"] = 1.0

        # Kad je neophodno izbeci hranu
        # kad moze slobodno da jede
        # foodList je lista pozicija hrana
        # x ->
        # y ^|
        if not features["normalGhosts"] or features["normalGhosts"] < 6:
            if food[newPosition[0]][newPosition[1]]:
                features["eatFood"] = 1.0
            if len(foodList) > 0:
                tempFood = []

                for food in foodList:
                    if (food[1] > 0) and (food[1] < walls.height / 3):
                        tempFood.append(food)

                if len(tempFood) == 0:
                    tempFood = foodList
                minDist = min([self.getMazeDistance(newPosition, food) for food in tempFood])
                if minDist is not None:
                    features["nearbyFood"] = float(minDist) / (walls.width * walls.height)

        features.divideAll(10.0)

        return features

    def getWeights(self, gameState, action):
        return {'eatInvader': 5, 'closeInvader': 0,'successorPosition' : -5,  'nearbyFood': -1, 'eatCapsule': 10.0,
                'normalGhosts': -20, 'eatGhost': 1.0, 'scaredGhosts': 0.1, 'stop': -5, 'eatFood': 1}


class DefensiveAgent(BaseAgent):

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
        features['numInvaders'] = len(invaders)

        if len(invaders) > 0:
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists)
        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1
        if (successor.getAgentState(self.index).scaredTimer > 0):
            features['numInvaders'] = 0
            if (features['invaderDistance'] <= 2): features['invaderDistance'] = 2
        teamNums = self.getTeam(gameState)
        initPos = gameState.getInitialAgentPosition(teamNums[0])
        # use the minimum noisy distance between our agent and their agent
        features['DistancefromStart'] = myPos[0] - initPos[0]
        if (features['DistancefromStart'] < 0): features['DistancefromStart'] *= -1
        if (features['DistancefromStart'] >= 10): features['DistancefromStart'] = 10
        if (features['DistancefromStart'] <= 4): features['DistancefromStart'] += 1
        if (features['DistancefromStart'] == 1):
            features['DistancefromStart'] == -9999
        features['DistancefromStart'] *= 2.5
        features['stayApart'] = self.getMazeDistance(gameState.getAgentPosition(teamNums[0]),
                                                     gameState.getAgentPosition(teamNums[1]))
        features['onDefense'] = 1
        features['offenseFood'] = 0

        if myState.isPacman:
            features['onDefense'] = -1

        if (len(invaders) == 0 and successor.getScore() != 0):
            features['onDefense'] = -1
            features['offenseFood'] = min(
                [self.getMazeDistance(myPos, food) for food in self.getFood(successor).asList()])
            features['foodCount'] = len(self.getFood(successor).asList())
            features['DistancefromStart'] = 0
            features['stayAprts'] += 2
            features['stayApart'] *= features['stayApart']
        if (len(invaders) != 0):
            features['stayApart'] = 0
            features['DistancefromStart'] = 0
        return features

    def getWeights(self, gameState, action):
        return {'foodCount': -20, 'offenseFood': -1, 'DistancefromStart': 3, 'numInvaders': -40000, 'onDefense': 20,
                'stayApart': 45, 'invaderDistance': -1800, 'stop': -400, 'reverse': -250}
