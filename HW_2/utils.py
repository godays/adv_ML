from scipy.sparse import vstack, hstack, csc_matrix
from scipy.stats.stats import spearmanr
from scipy.stats.stats import kendalltau
import numpy as np
import tqdm


def get_players_and_questions_qty(results_data, tournaments):
    players = set()
    for idx in tournaments.index:
        for team in results_data[idx]:
            team_id = team['team']['id']
            for player in team['teamMembers']:
                player_id = player['player']['id']
                players.add(player_id)
    return len(players), tournaments.total_questions.sum()


def sigmoid(x):
    x = 1.0 / (1.0 + np.exp(-1.0 * x))
    return x


def rating_adjustment(x):
    t = x / 100 - 5
    return sigmoid(t)


def adjust_rating(x):
    if x.total_questions > 1000:
        return x.skill_positive
    return x.skill_positive * rating_adjustment(x.total_questions)
