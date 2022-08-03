import math
from typing import Callable
from adversarialsearchproblem import (
    Action,
    AdversarialSearchProblem,
    State as GameState,
)

def minimax(asp: AdversarialSearchProblem[GameState, Action]) -> Action:
    player = asp.get_start_state().player_to_move()
    bestActionIndex = 0
    bestActionEvaluation = -math.inf
    actionList = list(asp.get_available_actions(asp.get_start_state()))
    for action in actionList:
        successorState = asp.transition(asp.get_start_state(), action)
        successorEvaluation = maxValue(asp, successorState, player)
        if (successorEvaluation > bestActionEvaluation):
            bestActionEvaluation = successorEvaluation
            bestActionIndex = actionList.index(action)
    return actionList[bestActionIndex]

def maxValue(asp: AdversarialSearchProblem, state, player):
    if (asp.is_terminal_state(state) == True):
        return asp.evaluate_terminal(state)[player]
    else:
        currentMax = -math.inf
        for action in asp.get_available_actions(state):
            successorState = asp.transition(state, action)
            currentEvaluation = minValue(asp, successorState, player)
            currentMax = max(currentMax, currentEvaluation)
        return currentMax

def minValue(asp: AdversarialSearchProblem, state, player):
    if (asp.is_terminal_state(state) == True):
        return asp.evaluate_terminal(state)[player]
    else:
        currentMin = math.inf
        for action in asp.get_available_actions(state):
            successorState = asp.transition(state, action)
            currentEvaluation = maxValue(asp, successorState, player)
            currentMin = min(currentMin, currentEvaluation)
        return currentMin

def alpha_beta(asp: AdversarialSearchProblem[GameState, Action]) -> Action:
    player = asp.get_start_state().player_to_move()
    bestActionIndex = 0
    bestActionEvaluation = -math.inf
    actionList = list(asp.get_available_actions(asp.get_start_state()))
    for action in actionList:
        successorState = asp.transition(asp.get_start_state(), action)
        alpha = -math.inf
        beta = math.inf
        successorEvaluation = maxValueAB(asp, alpha, beta, successorState, player)
        if (successorEvaluation > bestActionEvaluation):
            bestActionEvaluation = successorEvaluation
            bestActionIndex = actionList.index(action)
    return actionList[bestActionIndex]

def maxValueAB(asp: AdversarialSearchProblem, alpha, beta, state, player):
    if (asp.is_terminal_state(state) == True):
        return asp.evaluate_terminal(state)[player]
    else:
        currentMax = -math.inf
        for action in asp.get_available_actions(state):
            successorState = asp.transition(state, action)
            currentEvaluation = minValueAB(asp, alpha, beta, successorState, player)
            currentMax = max(currentMax, currentEvaluation)
            alpha = max(alpha, currentEvaluation)
            if (beta <= alpha):
                break
        return currentMax

def minValueAB(asp: AdversarialSearchProblem, alpha, beta, state, player):
    if (asp.is_terminal_state(state) == True):
        return asp.evaluate_terminal(state)[player]
    else:
        currentMin = math.inf
        for action in asp.get_available_actions(state):
            successorState = asp.transition(state, action)
            currentEvaluation = maxValueAB(asp, alpha, beta, successorState, player)
            currentMin = min(currentMin, currentEvaluation)
            beta = min(beta, currentEvaluation)
            if (beta <= alpha):
                break
        return currentMin

def alpha_beta_cutoff(asp: AdversarialSearchProblem[GameState, Action], cutoff_ply: int, heuristic_func: Callable[[GameState], float]) -> Action:
    player = asp.get_start_state().player_to_move()
    bestActionIndex = 0
    bestActionEvaluation = -math.inf
    actionList = list(asp.get_available_actions(asp.get_start_state()))
    for action in actionList:
        successorState = asp.transition(asp.get_start_state(), action)
        alpha = -math.inf
        beta = math.inf
        successorEvaluation = maxValueABC(asp, alpha, beta, cutoff_ply, successorState, player)
        if (successorEvaluation > bestActionEvaluation):
            bestActionEvaluation = successorEvaluation
            bestActionIndex = actionList.index(action)
    return actionList[bestActionIndex]
    

def maxValueABC(asp: AdversarialSearchProblem, alpha, beta, cutoff, state, player):
    if (asp.is_terminal_state(state) == True):
        return asp.evaluate_terminal(state)[player]
    elif (cutoff == 0):
        return asp.heuristic_func(state, player)
    else:
        currentMax = -math.inf
        for action in asp.get_available_actions(state):
            successorState = asp.transition(state, action)
            currentEvaluation = minValueABC(asp, alpha, beta, cutoff - 1, successorState, player)
            currentMax = max(currentMax, currentEvaluation)
            alpha = max(alpha, currentEvaluation)
            if (beta <= alpha):
                break
        return currentMax

def minValueABC(asp: AdversarialSearchProblem, alpha, beta, cutoff, state, player):
    if (asp.is_terminal_state(state) == True):
        return asp.evaluate_terminal(state)[player]
    elif (cutoff == 0):
        return asp.heuristic_func(state, player)
    else:
        currentMin = math.inf
        for action in asp.get_available_actions(state):
            successorState = asp.transition(state, action)
            currentEvaluation = maxValueABC(asp, alpha, beta, cutoff - 1, successorState, player)
            currentMin = min(currentMin, currentEvaluation)
            beta = min(beta, currentEvaluation)
            if (beta <= alpha):
                break
        return currentMin
