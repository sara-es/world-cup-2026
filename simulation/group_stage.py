"""Group stage simulation using Poisson goal model."""
import math
import numpy as np
import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class TeamRecord:
    """Track a team's group stage performance."""
    team: str
    points: int = 0
    goals_for: int = 0
    goals_against: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0

    @property
    def goal_diff(self) -> int:
        return self.goals_for - self.goals_against


def simulate_match(
    strength1: float,
    strength2: float,
    mu: float = math.log(1.3),
    alpha: float = 0.35,
    rng: np.random.Generator = np.random.default_rng()
) -> Tuple[int, int]:
    """
    Simulate a single match using Poisson goal model.

    Goals for each team follow independent Poisson distributions:
        lambda_i = exp(mu + alpha * (strength_i - strength_j))

    Args:
        strength1: Strength of team1 [0, 1]
        strength2: Strength of team2 [0, 1]
        mu: Base rate parameter (default: ln(1.3) ≈ 0.262)
        alpha: Strength effect parameter (default: 0.5)

    Returns:
        Tuple of (goals_team1, goals_team2)
    """
    # Calculate Poisson rate parameters
    lambda1 = math.exp(mu + alpha * (strength1 - strength2))
    lambda2 = math.exp(mu + alpha * (strength2 - strength1))

    # Generate goals
    goals1 = rng.poisson(lambda1)
    goals2 = rng.poisson(lambda2)

    return goals1, goals2


def update_record(record: TeamRecord, goals_for: int, goals_against: int) -> None:
    """
    Update team record based on match result.

    Args:
        record: TeamRecord to update (modified in place)
        goals_for: Goals scored by this team
        goals_against: Goals conceded by this team
    """
    record.goals_for += goals_for
    record.goals_against += goals_against

    if goals_for > goals_against:
        record.wins += 1
        record.points += 3
    elif goals_for == goals_against:
        record.draws += 1
        record.points += 1
    else:
        record.losses += 1


def simulate_group(
    teams: List[str],
    strengths: Dict[str, float],
    fifa_rankings: Dict[str, float],
    mu: float = math.log(1.3),
    alpha: float = 0.35, 
    rng: np.random.Generator = np.random.default_rng()
) -> List[TeamRecord]:
    """
    Simulate all matches in a group and return final standings.

    Args:
        teams: List of 4 team names
        strengths: Dict mapping team name to strength [0, 1]
        fifa_rankings: Dict mapping team name to FIFA ranking (for tiebreaker)
        mu: Base rate parameter
        alpha: Strength effect parameter

    Returns:
        List of TeamRecords sorted by final standings (1st, 2nd, 3rd, 4th)
    """
    # Initialize records
    records = {team: TeamRecord(team=team) for team in teams}

    # Play all matches (round-robin: 6 matches for 4 teams)
    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            team1, team2 = teams[i], teams[j]
            goals1, goals2 = simulate_match(
                strengths[team1], strengths[team2],
                mu, alpha, rng
            )

            update_record(records[team1], goals1, goals2)
            update_record(records[team2], goals2, goals1)

    # Sort by FIFA tiebreaker rules:
    # 1. Points
    # 2. Goal difference
    # 3. Goals scored
    # 4. FIFA ranking (higher rating = better)
    standings = sorted(
        records.values(),
        key=lambda r: (
            r.points,
            r.goal_diff,
            r.goals_for,
            fifa_rankings.get(r.team, 0)  # Higher FIFA rating is better
        ),
        reverse=True
    )

    return standings


def simulate_group_with_results(
    teams: List[str],
    strengths: Dict[str, float],
    fifa_rankings: Dict[str, float],
    completed_standings: Optional[Dict[str, Dict]] = None,
    remaining_matches: Optional[List[Tuple[str, str]]] = None,
    mu: float = math.log(1.3),
    alpha: float = 0.35,
    rng: np.random.Generator = np.random.default_rng()
) -> List[TeamRecord]:
    """
    Simulate a group with some completed matches.

    Args:
        teams: List of 4 team names
        strengths: Dict mapping team name to strength
        fifa_rankings: Dict mapping team name to FIFA ranking
        completed_standings: Current standings from completed matches (or None)
        remaining_matches: List of matches still to play (or None for all)
        mu: Base rate parameter
        alpha: Strength effect parameter
        rng: Random number generator

    Returns:
        List of TeamRecords sorted by final standings
    """
    # Initialize records from completed standings or from scratch
    records = {}
    if completed_standings:
        for team in teams:
            stats = completed_standings.get(team, {})
            records[team] = TeamRecord(
                team=team,
                points=stats.get('points', 0),
                goals_for=stats.get('goals_for', 0),
                goals_against=stats.get('goals_against', 0),
                wins=stats.get('wins', 0),
                draws=stats.get('draws', 0),
                losses=stats.get('losses', 0)
            )
    else:
        records = {team: TeamRecord(team=team) for team in teams}

    # Determine which matches to simulate
    if remaining_matches is None:
        # Simulate all matches
        matches_to_sim = [(teams[i], teams[j])
                          for i in range(len(teams))
                          for j in range(i + 1, len(teams))]
    else:
        matches_to_sim = remaining_matches

    # Simulate remaining matches
    for team1, team2 in matches_to_sim:
        goals1, goals2 = simulate_match(
            strengths[team1], strengths[team2],
            mu, alpha, rng
        )
        update_record(records[team1], goals1, goals2)
        update_record(records[team2], goals2, goals1)

    # Sort by FIFA tiebreaker rules
    standings = sorted(
        records.values(),
        key=lambda r: (
            r.points,
            r.goal_diff,
            r.goals_for,
            fifa_rankings.get(r.team, 0)
        ),
        reverse=True
    )

    return standings


def simulate_all_groups(
    groups: Dict[str, List[str]],
    strengths: Dict[str, float],
    fifa_rankings: Dict[str, float],
    mu: float = math.log(1.3),
    alpha: float = 0.35,
    rng: np.random.Generator = np.random.default_rng(),
    completed_matches = None
) -> Dict[str, List[TeamRecord]]:
    """
    Simulate all group stage matches, using real results if available.

    Args:
        groups: Dict mapping group letter to list of 4 teams
        strengths: Dict mapping team name to strength
        fifa_rankings: Dict mapping team name to FIFA ranking
        mu: Base rate parameter
        alpha: Strength effect parameter
        rng: Random number generator
        completed_matches: CompletedMatches instance (optional)

    Returns:
        Dict mapping group letter to sorted standings
    """
    all_standings = {}

    for group_letter, teams in groups.items():
        if completed_matches:
            # Use hybrid simulation with real results
            from simulation.match_loader import get_remaining_group_matches

            current_standings = completed_matches.get_group_standings(group_letter, teams)
            remaining = get_remaining_group_matches(group_letter, teams, completed_matches)

            standings = simulate_group_with_results(
                teams, strengths, fifa_rankings,
                current_standings, remaining,
                mu, alpha, rng
            )
        else:
            # Pure simulation
            standings = simulate_group(teams, strengths, fifa_rankings, mu, alpha, rng)

        all_standings[group_letter] = standings

    return all_standings


class GroupStageResult:
    """Container for group stage results."""

    def __init__(self, standings: Dict[str, List[TeamRecord]]):
        self.standings = standings

    def get_winners(self) -> Dict[str, str]:
        """Get group winners (1st place teams)."""
        return {group: teams[0].team for group, teams in self.standings.items()}

    def get_runners_up(self) -> Dict[str, str]:
        """Get group runners-up (2nd place teams)."""
        return {group: teams[1].team for group, teams in self.standings.items()}

    def get_third_place(self) -> Dict[str, TeamRecord]:
        """Get all third-place teams with their records."""
        return {group: teams[2] for group, teams in self.standings.items()}

    def get_all_advancing(self) -> List[str]:
        """Get all teams advancing from group stage (before third-place selection)."""
        advancing = []
        for teams in self.standings.values():
            advancing.extend([teams[0].team, teams[1].team, teams[2].team])
        return advancing

    def get_eliminated(self) -> List[str]:
        """Get all teams eliminated in group stage (4th place)."""
        return [teams[3].team for teams in self.standings.values()]
