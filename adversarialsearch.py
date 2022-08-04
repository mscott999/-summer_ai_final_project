import numpy
from typing import Callable
from adversarialsearchproblem import (Action, AdversarialSearchProblem, State as GameState)

def minimax(asp: AdversarialSearchProblem[GameState, Action]) -> Action:
    player = asp.get_start_state().player_to_move()
    if (player == 0):
        return maxValue(asp, asp.get_start_state())[0]
    else:
        return minValue(asp, asp.get_start_state())[0]

def maxValue(asp: AdversarialSearchProblem, state):
    output = [None, numpy.NINF]
    if (asp.is_terminal_state(state) == True):
        output[1] = asp.evaluate_terminal(state)[0]
        return output
    else:
        for action in asp.get_available_actions(state):
            successorState = asp.transition(state, action)
            currentEvaluation = minValue(asp, successorState)[1]
            if (currentEvaluation > output[1]):
                output = [action, currentEvaluation]
        return output

def minValue(asp: AdversarialSearchProblem, state):
    output = [None, numpy.Inf]
    if (asp.is_terminal_state(state) == True):
        output[1] = asp.evaluate_terminal(state)[0]
        return output
    else:
        for action in asp.get_available_actions(state):
            successorState = asp.transition(state, action)
            currentEvaluation = maxValue(asp, successorState)[1]
            if (currentEvaluation < output[1]):
                output = [action, currentEvaluation]
        return output

def alpha_beta(asp: AdversarialSearchProblem[GameState, Action]) -> Action:
    player = asp.get_start_state().player_to_move()
    if (player == 0):
        return maxValueAB(asp, asp.get_start_state(), numpy.NINF, numpy.Inf)[0]
    else:
        return minValueAB(asp, asp.get_start_state(), numpy.NINF, numpy.Inf)[0]

def maxValueAB(asp: AdversarialSearchProblem, state, alpha, beta):
    output = [None, numpy.NINF]
    if (asp.is_terminal_state(state) == True):
        output[1] = asp.evaluate_terminal(state)[0]
        return output
    else:
        for action in asp.get_available_actions(state):
            successorState = asp.transition(state, action)
            currentEvaluation = minValueAB(asp, successorState, alpha, beta)[1]
            if (currentEvaluation > output[1]):
                output = [action, currentEvaluation]
            alpha = max(alpha, currentEvaluation)
            if(alpha >= beta):
                break
        return output

def minValueAB(asp: AdversarialSearchProblem, state, alpha, beta):
    output = [None, numpy.Inf]
    if (asp.is_terminal_state(state) == True):
        output[1] = asp.evaluate_terminal(state)[0]
        return output
    else:
        for action in asp.get_available_actions(state):
            successorState = asp.transition(state, action)
            currentEvaluation = maxValueAB(asp, successorState, alpha, beta)[1]
            if (currentEvaluation < output[1]):
                output = [action, currentEvaluation]
            beta = min(beta, currentEvaluation)
            if(alpha >= beta):
                break
        return output

def alpha_beta_cutoff(asp: AdversarialSearchProblem[GameState, Action], cutoff_ply: int, heuristic_func: Callable[[GameState], float]) -> Action:
    player = asp.get_start_state().player_to_move()
    if (player == 0):
        return maxValueABC(asp, asp.get_start_state(), numpy.NINF, numpy.Inf, cutoff_ply)[0]
    else:
        return minValueABC(asp, asp.get_start_state(), numpy.NINF, numpy.Inf, cutoff_ply)[0]

def maxValueABC(asp: AdversarialSearchProblem, state, alpha, beta, cutoff):
    output = [None, numpy.NINF]
    if (asp.is_terminal_state(state) == True):
        output[1] = asp.evaluate_terminal(state)[0]
        return output
    elif (cutoff == 0):
        output[1] = asp.heuristic_func(state, 0)
        return output
    else:
        for action in asp.get_available_actions(state):
            successorState = asp.transition(state, action)
            currentEvaluation = minValueABC(asp, successorState, alpha, beta, cutoff - 1)[1]
            if (currentEvaluation > output[1]):
                output = [action, currentEvaluation]
            alpha = max(alpha, currentEvaluation)
            if(alpha >= beta):
                break
        return output

def minValueABC(asp: AdversarialSearchProblem, state, alpha, beta, cutoff):
    output = [None, numpy.Inf]
    if (asp.is_terminal_state(state) == True):
        output[1] = asp.evaluate_terminal(state)[0]
        return output
    elif (cutoff == 0):
        output[1] = asp.heuristic_func(state, 0)
        return output
    else:
        for action in asp.get_available_actions(state):
            successorState = asp.transition(state, action)
            currentEvaluation = maxValueABC(asp, successorState, alpha, beta, cutoff - 1)[1]
            if (currentEvaluation < output[1]):
                output = [action, currentEvaluation]
            beta = min(beta, currentEvaluation)
            if(alpha >= beta):
                break
        return output
