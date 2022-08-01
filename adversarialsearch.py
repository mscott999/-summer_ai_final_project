import math
from typing import Callable

from adversarialsearchproblem import (
    Action,
    AdversarialSearchProblem,
    State as GameState,
)


def minimax(asp: AdversarialSearchProblem[GameState, Action]) -> Action:
    """
    Implement the minimax algorithm on ASPs, assuming that the given game is
    both 2-player and constant-sum.
    Input:
        asp - an AdversarialSearchProblem
    Output:
        an action (an element of asp.get_available_actions(asp.get_start_state()))
    """
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
    """
    Implement the alpha-beta pruning algorithm on ASPs,
    assuming that the given game is both 2-player and constant-sum.

    Input:
        asp - an AdversarialSearchProblem
    Output:
        an action(an element of asp.get_available_actions(asp.get_start_state()))
    """
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

def alpha_beta_cutoff(
    asp: AdversarialSearchProblem[GameState, Action],
    cutoff_ply: int,
    # See AdversarialSearchProblem:heuristic_func
    heuristic_func: Callable[[GameState], float],
) -> Action:
    """
    This function should:
    - search through the asp using alpha-beta pruning
    - cut off the search after cutoff_ply moves have been made.

    Input:
        asp - an AdversarialSearchProblem
        cutoff_ply - an Integer that determines when to cutoff the search and
            use heuristic_func. For example, when cutoff_ply = 1, use
            heuristic_func to evaluate states that result from your first move.
            When cutoff_ply = 2, use heuristic_func to evaluate states that
            result from your opponent's first move. When cutoff_ply = 3 use
            heuristic_func to evaluate the states that result from your second
            move. You may assume that cutoff_ply > 0.
        heuristic_func - a function that takes in a GameState and outputs a
            real number indicating how good that state is for the player who is
            using alpha_beta_cutoff to choose their action. You do not need to
            implement this function, as it should be provided by whomever is
            calling alpha_beta_cutoff, however you are welcome to write
            evaluation functions to test your implemention. The heuristic_func
            we provide does not handle terminal states, so evaluate terminal
            states the same way you evaluated them in the previous algorithms.
    Output:
        an action(an element of asp.get_available_actions(asp.get_start_state()))
    """
    ...
