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
        dist = self.getMazeDistance(self.start,pos2)
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
  """
  Our idea of an offensive agent implementation.
  """
  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    # Required data
    allFood = self.getFood(gameState)
    allCapsules = gameState.getCapsules()
    foodList = self.getFood(successor).asList()
    #features['successorScore'] = -len(foodList)#self.getScore(successor)
    walls = gameState.getWalls()
    myPos = gameState.getAgentState(self.index).getPosition()
    vector = Actions.directionToVector(action)
    newPos = (int(myPos[0] + vector[0]), int(myPos[1] + vector[1]))

    # Offenders (pacmans) and defenders(ghosts)
    enemies = [gameState.getAgentState(a) for a in self.getOpponents(gameState)]
    offenders = [a for a in enemies if a.isPacman and a.getPosition() is not None]
    defenders = [a for a in enemies if not a.isPacman and a.getPosition() is not None]

    # If pacman has stopped
    if action == Directions.STOP:
        features['stop'] = 1.0

    # Capsules
    for cPos in allCapsules:
        if newPos == cPos and successor.getAgentState(self.index).isPacman:
            features['eatCapsule'] = 1.0

    # Ghosts
    for ghost in defenders:
        ghostPosition = ghost.getPosition()
        ghostNeighbors = Actions.getLegalNeighbors(ghostPosition, walls)
        if newPos == ghostPosition and ghost.scaredTimer > 0:
            features['eatGhost'] += 1
            features['eatFood'] += 2
        elif newPos == ghostPosition and ghost.scaredTimer == 0:
            features['scaredGhost'] = 0
            features['normalGhost'] = 1
        elif newPos in ghostNeighbors and ghost.scaredTimer > 0:
            features['scaredGhost'] += 1
        elif successor.getAgentState(self.index).isPacman and ghost.scaredTimer > 0:
            features['scaredGhosts'] = 0
            features['normalGhosts'] += 1

    # Scared or not scared
    # for pacman in offenders:
    #     pacmanPos = pacman.getPosition()
    #     pacmanNeighbors = Actions.getLegalNeighbors(pacmanPos, walls)
    #     if newPos == pacmanPos and gameState.getAgentState(self.index).scaredTimer == 0:
    #         features['eatOffender'] = 1
    #     elif newPos in pacmanNeighbors and gameState.getAgentState(self.index).scaredTimer == 0:
    #         features['closeOffender'] += 1
    #     elif newPos == pacmanPos and gameState.getAgentState(self.index).scaredTimer > 0:
    #         features['eatOffender'] = -10
    #     elif newPos in pacmanNeighbors and gameState.getAgentState(self.index).scaredTimer > 0:
    #         features['eatOffender'] = -10
    #         features['closeOffender'] += -10

    if gameState.getAgentState(self.index).scaredTimer == 0:
      # ghost je pacman????
      for ghost in offenders:
        ghostpos = ghost.getPosition()
        neighbors = Actions.getLegalNeighbors(ghostpos, walls)
        if newPos == ghostpos:
          features['eatOffender'] = 1
        elif newPos in neighbors:
          features['closeOffender'] += 1
    # uplasen
    else:
      for ghost in enemies:
        if ghost.getPosition() != None:
          ghostpos = ghost.getPosition()
          neighbors = Actions.getLegalNeighbors(ghostpos,
                                                walls)
          if newPos in neighbors:
            features['closeOffender'] += -10
            features['eatOffender'] = -10
          elif newPos == ghostpos:
            features['eatOffender'] = -10


    # Compute distance to the nearest food
    # if len(foodList) > 0:
    #   if allFood[newPos[0]][newPos[1]]:
    #       features['eatFood'] = 1.0
    #   myPos = successor.getAgentState(self.index).getPosition()
    #   minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
    #   features['distanceToFood'] = minDistance

    if not features['normalGhosts']:
      if allFood[newPos[0]][newPos[1]]:
        features['eatFood'] = 1.0
      if len(foodList) > 0:
        tempFood = []
        for food in foodList:
          (food_x, food_y) = food
          adjustedindex = self.index - self.index % 2
          check1 = food_y > adjustedindex / 2 * walls.height / 3
          check2 = food_y < (adjustedindex / 2 + 1) * walls.height / 3
          if check1 and check2:
            tempFood.append(food)
        if len(tempFood) == 0:
          tempFood = foodList
        mazedist = [self.getMazeDistance(newPos, food)
                    for food in tempFood]
        if min(mazedist) is not None:
          walldimensions = walls.width * walls.height
          features['distanceToFood'] = float(min(mazedist))/ walldimensions
    return features

  def getWeights(self, gameState, action):
      return {'eatOffender': 5,
              'closeOffender': 0,
              'distanceToFood': -1,
              'eatCapsule': 10.0,
              'normalGhost': -20,
              'eatGhost': 1.0,
              'scaredGhost': 0.1,
              'stop': -5,
              'eatFood': 1.0}

class DefensiveAgent(BaseAgent):
   pass