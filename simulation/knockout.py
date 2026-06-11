"""Knockout stage simulation using Bradley-Terry model."""
import random
from typing import Dict, List, Tuple


def bradley_terry_win_prob(
    strength_i: float,
    strength_j: float,
    k: float = 0.5
) -> float:
    """
    Calculate win probability for team i vs team j using Bradley-Terry model.

    P(i beats j) = strength_i^k / (strength_i^k + strength_j^k)

    Args:
        strength_i: Strength of team i [0, 1]
        strength_j: Strength of team j [0, 1]
        k: Exponent parameter (default: 0.5, higher = more deterministic)

    Returns:
        Probability that team i beats team j
    """
    # Add small epsilon to avoid division by zero
    eps = 1e-10
    strength_i = max(strength_i, eps)
    strength_j = max(strength_j, eps)

    si_k = strength_i ** k
    sj_k = strength_j ** k

    return si_k / (si_k + sj_k)


def simulate_knockout_match(
    team1: str,
    team2: str,
    strength1: float,
    strength2: float,
    k: float = 0.5
) -> str:
    """
    Simulate a single knockout match (no draws).

    Args:
        team1: First team name
        team2: Second team name
        strength1: Strength of team1 [0, 1]
        strength2: Strength of team2 [0, 1]
        k: Bradley-Terry exponent (default: 0.5)

    Returns:
        Name of winning team
    """
    win_prob = bradley_terry_win_prob(strength1, strength2, k)

    if random.random() < win_prob:
        return team1
    else:
        return team2


def simulate_knockout_round(
    matchups: List[Tuple[str, str]],
    strengths: Dict[str, float],
    k: float = 0.5
) -> List[str]:
    """
    Simulate a complete knockout round.

    Args:
        matchups: List of (team1, team2) tuples
        strengths: Dict mapping team name to strength
        k: Bradley-Terry exponent

    Returns:
        List of winning teams
    """
    winners = []

    for team1, team2 in matchups:
        winner = simulate_knockout_match(
            team1, team2,
            strengths[team1], strengths[team2],
            k
        )
        winners.append(winner)

    return winners


def build_next_round_matchups(winners: List[str]) -> List[Tuple[str, str]]:
    """
    Build matchups for next round by pairing winners sequentially.

    Args:
        winners: List of n winning teams

    Returns:
        List of n/2 matchup tuples
    """
    matchups = []
    for i in range(0, len(winners), 2):
        matchups.append((winners[i], winners[i + 1]))
    return matchups


class KnockoutStage:
    """
    Simulate complete knockout stage from R32 to Final.

    Stages:
    - R32: 32 teams → 16 teams (16 matches)
    - R16: 16 teams → 8 teams (8 matches)
    - QF: 8 teams → 4 teams (4 matches)
    - SF: 4 teams → 2 teams (2 matches)
    - Final: 2 teams → 1 champion (1 match)
    """

    def __init__(
        self,
        r32_matchups: List[Tuple[str, str]],
        strengths: Dict[str, float],
        k: float = 0.5
    ):
        """
        Initialize knockout stage.

        Args:
            r32_matchups: List of 16 R32 matchup tuples
            strengths: Dict mapping team name to strength
            k: Bradley-Terry exponent
        """
        self.r32_matchups = r32_matchups
        self.strengths = strengths
        self.k = k

        # Track results at each stage
        self.r32_winners = []
        self.r16_winners = []
        self.qf_winners = []
        self.sf_winners = []
        self.champion = None

        self.r16_matchups = []
        self.qf_matchups = []
        self.sf_matchups = []

    def simulate(self) -> str:
        """
        Simulate entire knockout stage.

        Returns:
            Name of champion team
        """
        # R32: 16 matches
        self.r32_winners = simulate_knockout_round(
            self.r32_matchups, self.strengths, self.k
        )

        # R16: 8 matches
        self.r16_matchups = build_next_round_matchups(self.r32_winners)
        self.r16_winners = simulate_knockout_round(
            self.r16_matchups, self.strengths, self.k
        )

        # Quarterfinals: 4 matches
        self.qf_matchups = build_next_round_matchups(self.r16_winners)
        self.qf_winners = simulate_knockout_round(
            self.qf_matchups, self.strengths, self.k
        )

        # Semifinals: 2 matches
        self.sf_matchups = build_next_round_matchups(self.qf_winners)
        self.sf_winners = simulate_knockout_round(
            self.sf_matchups, self.strengths, self.k
        )

        # Final: 1 match
        final_matchup = [(self.sf_winners[0], self.sf_winners[1])]
        champions = simulate_knockout_round(
            final_matchup, self.strengths, self.k
        )
        self.champion = champions[0]

        return self.champion

    def get_eliminated_by_stage(self) -> Dict[str, List[str]]:
        """
        Get teams eliminated at each knockout stage.

        Returns:
            Dict mapping stage name to list of eliminated teams
        """
        # Extract losers from each round
        def get_losers(matchups, winners):
            losers = []
            for team1, team2 in matchups:
                if team1 not in winners:
                    losers.append(team1)
                else:
                    losers.append(team2)
            return losers

        r32_losers = get_losers(self.r32_matchups, self.r32_winners)
        r16_losers = get_losers(self.r16_matchups, self.r16_winners)
        qf_losers = get_losers(self.qf_matchups, self.qf_winners)
        sf_losers = get_losers(self.sf_matchups, self.sf_winners)

        # Final loser (runner-up)
        final_loser = [t for t in self.sf_winners if t != self.champion]

        return {
            'R32': r32_losers,
            'R16': r16_losers,
            'QF': qf_losers,
            'SF': sf_losers,
            'Final': final_loser
        }
