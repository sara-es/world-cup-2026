"""Group stage simulation using Poisson goal model."""
import math
import numpy as np
import random
from functools import cmp_to_key
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


def _mini_league_stats(
    tier: List['TeamRecord'],
    match_results: Dict[Tuple[str, str], Tuple[int, int]]
) -> Dict[str, Dict[str, int]]:
    """Compute points/GD/GF among a subset of teams using only their head-to-head matches."""
    tier_set = {t.team for t in tier}
    stats = {t.team: {'pts': 0, 'gd': 0, 'gf': 0} for t in tier}
    for (t1, t2), (g1, g2) in match_results.items():
        if t1 in tier_set and t2 in tier_set:
            stats[t1]['gf'] += g1
            stats[t1]['gd'] += g1 - g2
            stats[t2]['gf'] += g2
            stats[t2]['gd'] += g2 - g1
            if g1 > g2:
                stats[t1]['pts'] += 3
            elif g1 == g2:
                stats[t1]['pts'] += 1
                stats[t2]['pts'] += 1
            else:
                stats[t2]['pts'] += 3
    return stats


def rank_group(
    records: Dict[str, 'TeamRecord'],
    match_results: Dict[Tuple[str, str], Tuple[int, int]],
    fair_play: Dict[str, int],
    fifa_rankings: Dict[str, float]
) -> List['TeamRecord']:
    """
    Rank teams using official FIFA 2026 World Cup tiebreaker rules:
    1. Points
    2. Mini-league (head-to-head): points, then GD, then GF among tied teams
    3. Overall GD, overall GF
    4. Fair play record (higher = better)
    5. FIFA ranking (higher = better)
    """
    all_teams = list(records.values())
    by_points = sorted(all_teams, key=lambda r: r.points, reverse=True)

    result: List['TeamRecord'] = []
    i = 0
    while i < len(by_points):
        j = i + 1
        while j < len(by_points) and by_points[j].points == by_points[i].points:
            j += 1
        tier = by_points[i:j]

        if len(tier) == 1:
            result.extend(tier)
        else:
            mini = _mini_league_stats(tier, match_results)

            def cmp(a: 'TeamRecord', b: 'TeamRecord') -> int:
                for key in ('pts', 'gd', 'gf'):
                    if mini[a.team][key] != mini[b.team][key]:
                        return mini[b.team][key] - mini[a.team][key]
                if a.goal_diff != b.goal_diff:
                    return b.goal_diff - a.goal_diff
                if a.goals_for != b.goals_for:
                    return b.goals_for - a.goals_for
                fp_a = fair_play.get(a.team, 0)
                fp_b = fair_play.get(b.team, 0)
                if fp_a != fp_b:
                    return fp_b - fp_a
                r_a = fifa_rankings.get(a.team, 0)
                r_b = fifa_rankings.get(b.team, 0)
                return int((r_b - r_a) * 1000)

            result.extend(sorted(tier, key=cmp_to_key(cmp)))
        i = j

    return result


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
    rng: np.random.Generator = np.random.default_rng(),
    fair_play: Optional[Dict[str, int]] = None
) -> List[TeamRecord]:
    """
    Simulate all matches in a group and return final standings.

    Args:
        teams: List of 4 team names
        strengths: Dict mapping team name to strength [0, 1]
        fifa_rankings: Dict mapping team name to FIFA ranking (for tiebreaker)
        mu: Base rate parameter
        alpha: Strength effect parameter
        fair_play: Optional fair play ratings per team (for tiebreaking)

    Returns:
        List of TeamRecords sorted by final standings (1st, 2nd, 3rd, 4th)
    """
    records = {team: TeamRecord(team=team) for team in teams}
    match_results: Dict[Tuple[str, str], Tuple[int, int]] = {}

    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            team1, team2 = teams[i], teams[j]
            goals1, goals2 = simulate_match(
                strengths[team1], strengths[team2],
                mu, alpha, rng
            )
            update_record(records[team1], goals1, goals2)
            update_record(records[team2], goals2, goals1)
            match_results[(team1, team2)] = (goals1, goals2)

    return rank_group(records, match_results, fair_play or {}, fifa_rankings)


def simulate_group_with_results(
    teams: List[str],
    strengths: Dict[str, float],
    fifa_rankings: Dict[str, float],
    completed_standings: Optional[Dict[str, Dict]] = None,
    remaining_matches: Optional[List[Tuple[str, str]]] = None,
    mu: float = math.log(1.3),
    alpha: float = 0.35,
    rng: np.random.Generator = np.random.default_rng(),
    completed_match_results: Optional[Dict[Tuple[str, str], Tuple[int, int]]] = None,
    fair_play: Optional[Dict[str, int]] = None
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
        completed_match_results: Individual results from completed matches, used for head-to-head tiebreaking
        fair_play: Optional fair play ratings per team

    Returns:
        List of TeamRecords sorted by final standings
    """
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

    match_results: Dict[Tuple[str, str], Tuple[int, int]] = dict(completed_match_results or {})

    if remaining_matches is None:
        matches_to_sim = [(teams[i], teams[j])
                          for i in range(len(teams))
                          for j in range(i + 1, len(teams))]
    else:
        matches_to_sim = remaining_matches

    for team1, team2 in matches_to_sim:
        goals1, goals2 = simulate_match(
            strengths[team1], strengths[team2],
            mu, alpha, rng
        )
        update_record(records[team1], goals1, goals2)
        update_record(records[team2], goals2, goals1)
        match_results[(team1, team2)] = (goals1, goals2)

    return rank_group(records, match_results, fair_play or {}, fifa_rankings)


def simulate_all_groups(
    groups: Dict[str, List[str]],
    strengths: Dict[str, float],
    fifa_rankings: Dict[str, float],
    mu: float = math.log(1.3),
    alpha: float = 0.35,
    rng: np.random.Generator = np.random.default_rng(),
    completed_matches=None,
    fair_play: Optional[Dict[str, int]] = None
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
                mu, alpha, rng,
                completed_match_results=completed_matches.get_group_match_results(group_letter),
                fair_play=fair_play
            )
        else:
            # Pure simulation
            standings = simulate_group(teams, strengths, fifa_rankings, mu, alpha, rng, fair_play=fair_play)

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
