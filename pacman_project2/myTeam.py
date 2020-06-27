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
from scipy.constants import value

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

        values = [self.evaluate(gameState, a) for a in actions]

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

    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)

        self.epsilon = 0.0  # exploration prob
        self.alpha = 0.2  # learning rate
        self.discountRate = 0.8
        self.weights = self.getWeights(gameState, None)

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
        """
        features = self.getFeatures(state, action)
        return features * self.weights

    def chooseAction(self, gameState):
        """
          MinMax ne zato sto necemo da uvek bira min ili max, nekad je bolji i prosek. Takodje
        performanse su slicne performansama DFS-a koji prolazi celo stablo
        Alfa-Beta odsecanje ne, neke korake cemo odseci a mozda su bili dobri, who will know
        
        Expectimax je ok kada ne znamo ishod akcije koju izvrsavamo. Ovde duhovi igraju sa
        odredjenom dozom slucajnosti. Kod Expectimax algoritma vrednosti u stanjima treba da 
        reprezentuju ishod koji se desava u proseku, a ne u najgorem slucaju (kao kod Minimaxa). Expectimax
        pretraa racuna prosecnu vrednost ako podazumevamo optimalno igranje. Max cvorovi su isti kao kod Minimax pretrage. Cvorovi slucajeva su kao MIN, 
        cvorovi, kod njih ishod nije siguran. Izracunavamo njihove ocekivane vrednosti (expected utillities), tj. u opstem
        slucaju racunamo prosek vrednosti njihovih potomaka. Expectimax
        uzima ocekivanu vrednost u odnosu na to kako se duhovi ponasaju. 
           
        Expectimax algoritam koriscen
          
        """
        actions = gameState.getLegalActions(self.index)
        values = [self.evaluate(gameState, a) for a in actions]
        pacman = gameState.getAgentState(self.index)
        bestActions = []
        # radimo pred korak samo ukoliko je pacman
        if pacman.isPacman:
            maxValue = self.max_value(gameState)
            bestActions = [a for a, v in zip(actions, values) if v == maxValue]
        else:
            maxValue = max(values)
            bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        # nemamo kad dodje do granice da se prebaci na svom terenu zbog poena i onda da se vrati
        # da gleda da li mu preprecava put duh, ako mu prepreci onda da se vraca unazad
        # i ako ima manje od 3 tackice hrane i ako su one blizu da ide ka njima, a ne na svojoj teritoriji

        # da ne treba da bezi od duhova ako je scared time

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

    def max_value(self, gameState):
        maximum = float("-inf")
        # ovo kad bi moglo kroz sve moguce akcije
        # ovo su akcije za trenutno stanje
        actions = gameState.getLegalActions(self.index)
        for action in actions:
           maximum = max(maximum,self.evaluate(gameState, action))
        return maximum

    def getFeatures(self, gameState, action):
        # potrebni podaci

        features = util.Counter()
        foodList = self.getFood(gameState).asList()
        capsules = gameState.getCapsules()
        walls = gameState.getWalls()

        currentPosition = gameState.getAgentState(self.index).getPosition()
        vector = Actions.directionToVector(action)
        newPosition = (int(currentPosition[0] + vector[0]), int(currentPosition[1] + vector[1]))

        enemies = [gameState.getAgentState(a) for a in self.getOpponents(gameState)]
        ghosts = [a for a in enemies if not a.isPacman and a.getPosition() is not None]
        pacmans = [a for a in enemies if a.isPacman and a.getPosition() is not None]


        if action == Directions.STOP:
            features["stop"] = 1

        successor = self.getSuccessor(gameState, action)
        # ako smo pacman, kako se ophoditi prema duhu
        self.as_pacman(features, ghosts, newPosition, walls, successor)

        notScared = gameState.getAgentState(self.index).scaredTimer == 0

        # kroz sve, prodji ovo opet
        for i in enemies:
            if i.getPosition() is not None:
                if newPosition in Actions.getLegalNeighbors(i.getPosition(), walls) and not notScared:
                    features["distanceToInvader"] += -10
                    features["eatInvader"] = -10
                    features["stop"] = 0
                elif newPosition == i.getPosition() and not notScared:
                    features["eatInvader"] = -10
                    features["stop"] = 0

        # ukoliko smo duh i idemo ka pacmanima
        self.as_ghost(features, pacmans, newPosition, notScared, walls)

        for c in capsules:
            if newPosition == c and successor.getAgentState(self.index).isPacman:
                features["eatCapsule"] = 2

            elif newPosition in Actions.getLegalNeighbors(c, walls) and successor.getAgentState(self.index).isPacman:
                features["eatCapsule"] = 1


        # ako je pacman za hranu
        # proveri ovo 8 sa onim koliko ga mnozis i sabiras
        self.eat_food(features, gameState, newPosition, foodList, walls)

        features.divideAll(10.0)

        return features

    def getWeights(self, gameState, action):
        return {
                 'stop': -5.0,
                 # ukoliko smo pacman
                 'normalGhostDistance': -20.0, # alarmantno
                 'scaredGhostDistance': -6.0,
                 'distanceToFood': -4.0,
                 'eatGhost': -1.0,
                 'eatFood': 1.0,
                 'eatCapsule': 10.0,
                 # ukoliko smo duh
                 'eatInvader': 9.0,
                 'distanceToInvader': 0,
                 'scared': -20.0,
                 # oba
                 'successorPosition': -5.0,
                 'reverse': 1.0
                }

    def as_pacman(self, features, ghosts, newPosition, walls, successor):
        for i in ghosts:
            # ukoliko su uplaseni duhovi  i ukoliko su na tacnoj poziciji
            # jedi ih
            if (i.scaredTimer > 0) and (newPosition == i.getPosition()):
                features['eatGhost'] += 5
                features['eatFood'] = 0
                features["distanceToFood"] = 1
                features["normalGhostDistance"] = 0
                features["scaredGhostDistance"] = 0
                features['successorPosition'] = - 9
                features["stop"] = 0
            # ukoliko je preplaseni duh u blizini, a vreme uplasenosti ne istice uskoro
            elif (i.scaredTimer > 2) and \
                    (newPosition in Actions.getLegalNeighbors(i.getPosition(), walls)):

                features["distanceToFood"] += 1
                features["scaredGhostDistance"] += 5
                features["normalGhostDistance"] = 0
                features['eatGhost'] = 0
                features['eatFood'] += 1  # usput ako mozes hranu da jedes dok juris za duhom
                features['successorPosition'] = - 9
                features["stop"] = 0

            # ukoliko je scared time manje od 2 i ako imamo preplasene blizu da bezi
            elif (i.scaredTimer < 3) and (newPosition in Actions.getLegalNeighbors(i.getPosition(), walls)):
                # beziiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii
                features["distanceToFood"] = 0
                features["normalGhostDistance"] += 5
                features["scaredGhostDistance"] = 0
                features["eatCapsule"] = 1
                features['eatGhost'] = 0
                features['eatFood'] = 0
                features["reverse"] += 2
                features['successorPosition'] += 50

            # ukoliko smo pacman i duhovi su normalni sta raditiiiiiiii
            if (i.scaredTimer == 0) and (newPosition == i.getPosition()):
                features["distanceToFood"] = 0
                features["normalGhostDistance"] += 10
                features["scaredGhostDistance"] = 0
                features["reverse"] = 0.0
                features["eatCapsule"] = 1
                features['eatGhost'] = 0
                features['eatFood'] = 0

                if len(successor.getLegalActions(self.index)) < 3:
                    features['successorPosition'] += 50

            elif (i.scaredTimer == 0) and (
                    newPosition in Actions.getLegalNeighbors(i.getPosition(), walls)):  # neprijatelj u blizini
                features["distanceToFood"] = 0
                features["normalGhostDistance"] += 5
                features["reverse"] = 0
                features["scaredGhostDistance"] = 0
                features["eatCapsule"] = 1
                features['eatGhost'] = 0
                features['eatFood'] = 0

                if len(successor.getLegalActions(self.index)) < 4:  # kako ga neprijatelj ne bi oterao u cosak
                    features['successorPosition'] += 50

    def as_ghost(self, features, pacmans, newPosition, notScared, walls):
        for i in pacmans:

            # ako nismo uplaseni i next pos je pacman
            if newPosition == i.getPosition() and notScared:

                features["eatInvader"] += 5
                features["distanceToInvader"] = 1
                features["scared"] = 0
                features["stop"] = 0
                features["reverse"] = 0

            # ako nismo uplaseni i pacman je blizu
            elif newPosition in Actions.getLegalNeighbors(i.getPosition(), walls) and notScared:
                features["eatInvader"] = 1
                features["distanceToInvader"] += 10
                features["scared"] = 0
                features["reverse"] = 0
            # ako smo uplaseni
            elif (newPosition in Actions.getLegalNeighbors(i.getPosition(), walls) or newPosition == i.getPosition()
            ) and (not notScared):

                features["eatInvader"] = 0
                features["distanceToInvader"] = 0
                features["scared"] += 5
                features["reverse"] += 10

    def eat_food(self, features, gameState, newPosition, foodList, walls):
        if not features["normalGhostDistance"] or features["normalGhostDistance"] < 6:
            if self.getFood(gameState)[newPosition[0]][newPosition[1]]:
                features["eatFood"] += 15
                features["distanceToFood"] = 0
                features["normalGhostDistance"] = 0
                features["reverse"] = 0
                features["scaredGhostDistance"] = 0
                features["eatCapsule"] = 0
                features['eatGhost'] = 0

            if len(foodList) > 0:
                tempFood = []

                for food in foodList:
                    if (food[1] > 0) and (food[1] < walls.height / 3):
                        tempFood.append(food)

                if len(tempFood) == 0:
                    tempFood = foodList
                minDist = min([self.getMazeDistance(newPosition, food) for food in tempFood])
                if minDist is not None:
                    features["distanceToFood"] = float(minDist) / (walls.width * walls.height)


class DefensiveAgent(BaseAgent):

    def chooseAction(self, gameState):
        """
          MinMax ne zato sto necemo da uvek bira min ili max, nekad je bolji i prosek. Takodje
        performanse su slicne performansama DFS-a koji prolazi celo stablo
        Alfa-Beta odsecanje ne, neke korake cemo odseci a mozda su bili dobri, who will know

        Expectimax je ok kada ne znamo ishod akcije koju izvrsavamo. Ovde duhovi igraju sa
        odredjenom dozom slucajnosti. Kod Expectimax algoritma vrednosti u stanjima treba da
        reprezentuju ishod koji se desava u proseku, a ne u najgorem slucaju (kao kod Minimaxa). Expectimax
        pretraa racuna prosecnu vrednost ako podazumevamo optimalno igranje. Max cvorovi su isti kao kod Minimax pretrage. Cvorovi slucajeva su kao MIN,
        cvorovi, kod njih ishod nije siguran. Izracunavamo njihove ocekivane vrednosti (expected utillities), tj. u opstem
        slucaju racunamo prosek vrednosti njihovih potomaka. Expectimax
        uzima ocekivanu vrednost u odnosu na to kako se duhovi ponasaju.

        Expectimax algoritam koriscen

        """

        actions = gameState.getLegalActions(self.index)
        values = [self.evaluate(gameState, a) for a in actions]
        pacman = gameState.getAgentState(self.index)

        # radimo pred korak samo ukoliko je pacman
        if pacman.isPacman:
            maxValue = max(values)
            bestActions = [a for a, v in zip(actions, values) if v == maxValue]
        else:
            approxValue, minimum = self.approx_find(gameState)
            bestActions = [a for a, v in zip(actions, values) if (approxValue <= v) and (v >= minimum)]

        # juri hranu juri hranu na vecoj udaljenosti, 15 koraka npr

        # foodLeft = len(self.getFood(gameState).asList())
        #
        # if foodLeft <= 2:
        #     bestDist = 9999
        #     for action in actions:
        #         successor = self.getSuccessor(gameState, action)
        #         pos2 = successor.getAgentPosition(self.index)
        #         dist = self.getMazeDistance(self.start, pos2)
        #         if dist < bestDist:
        #             bestAction = action
        #             bestDist = dist
        #     return bestAction

        return random.choice(bestActions)

    def approx_find(self, gameState):
        aprox = 0
        minimum = float("inf")
        actions = gameState.getLegalActions(self.index)
        for action in actions:
            aprox += self.evaluate(gameState, action)
            minimum = min(minimum, self.evaluate(gameState, action))
        return aprox / len(actions), minimum


    def getFeatures(self, gameState, action):
        # potreni podaci
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() is not None]

        features['numInvaders'] = len(invaders)

        if len(invaders) > 0:
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists)

        if action == Directions.STOP:
            features['stop'] = 1
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        if len(invaders) == 0:
            # ovde je prazno nek ide ka sredinii
            features['distanceFromEdge'] = 3
         # ukoliko smo preplaseni i juri nas pacman

        if myState.scaredTimer > 0:
            features['numInvaders'] = 0
            features['ghostsAreScared'] = 100
            features['invaderDistance'] = -2

        if myState.isPacman:
            features['onDefense'] = -1
        else:
            features['onDefense'] = 1

        return features

    def getWeights(self, gameState, action):
        return {'numInvaders': -40000,
                'onDefense': 20,
                'invaderDistance': -1800,
                'stop': -400,
                'reverse': -250, # -2
                'foodCount': -20,
                'offenseFood': -1,
                'ghostsAreScared': -1,
                'distanceFromEdge': -1}
