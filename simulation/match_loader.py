"""Load and process completed match results."""
import csv
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass


@dataclass
class MatchResult:
    """Represents a completed match result."""
    date: str
    stage: str
    group: Optional[str]
    team1: str
    team2: str
    team1_goals: int
    team2_goals: int
    team1_yellows: int
    team2_yellows: int
    team1_reds: int
    team2_reds: int
    extra_time: bool
    penalties: bool
    winner: Optional[str]

    @property
    def is_draw(self) -> bool:
        """Check if match was a draw (only possible in group stage)."""
        return self.winner is None or self.winner == ''

    def get_points(self, team: str) -> int:
        """Get points earned by team in this match."""
        if self.stage != 'Group':
            return 0  # Points only matter in group stage

        if self.is_draw:
            return 1
        elif self.winner == team:
            return 3
        else:
            return 0

    def get_goals_for(self, team: str) -> int:
        """Get goals scored by team."""
        if team == self.team1:
            return self.team1_goals
        elif team == self.team2:
            return self.team2_goals
        else:
            raise ValueError(f"Team {team} not in this match")

    def get_goals_against(self, team: str) -> int:
        """Get goals conceded by team."""
        if team == self.team1:
            return self.team2_goals
        elif team == self.team2:
            return self.team1_goals
        else:
            raise ValueError(f"Team {team} not in this match")

    def get_disciplinary_points(self, team: str) -> int:
        """
        Get disciplinary points (negative = worse).
        FIFA rules: Yellow = -1, Indirect red (2 yellows) = -3, Direct red = -4
        For simplicity: we'll use -1 per yellow, -4 per red.
        """
        if team == self.team1:
            return -(self.team1_yellows + 4 * self.team1_reds)
        elif team == self.team2:
            return -(self.team2_yellows + 4 * self.team2_reds)
        else:
            raise ValueError(f"Team {team} not in this match")


class CompletedMatches:
    """Container for all completed matches."""

    def __init__(self, matches: List[MatchResult]):
        self.matches = matches
        self._by_stage = self._group_by_stage()
        self._by_group = self._group_by_group()

    def _group_by_stage(self) -> Dict[str, List[MatchResult]]:
        """Group matches by stage."""
        by_stage = {}
        for match in self.matches:
            if match.stage not in by_stage:
                by_stage[match.stage] = []
            by_stage[match.stage].append(match)
        return by_stage

    def _group_by_group(self) -> Dict[str, List[MatchResult]]:
        """Group matches by group letter (group stage only)."""
        by_group = {}
        for match in self.matches:
            if match.stage == 'Group' and match.group:
                if match.group not in by_group:
                    by_group[match.group] = []
                by_group[match.group].append(match)
        return by_group

    def get_group_matches(self, group: str) -> List[MatchResult]:
        """Get all completed matches for a group."""
        return self._by_group.get(group, [])

    def get_stage_matches(self, stage: str) -> List[MatchResult]:
        """Get all completed matches for a stage."""
        return self._by_stage.get(stage, [])

    def get_teams_with_matches(self, group: str) -> Set[str]:
        """Get set of teams that have played at least one match in a group."""
        teams = set()
        for match in self.get_group_matches(group):
            teams.add(match.team1)
            teams.add(match.team2)
        return teams

    def has_group_completed(self, group: str) -> bool:
        """Check if all group stage matches are complete (6 matches for 4 teams)."""
        return len(self.get_group_matches(group)) == 6

    def get_group_standings(self, group: str, all_teams: List[str]) -> Dict[str, Dict]:
        """
        Get current standings for a group based on completed matches.

        Returns:
            Dict mapping team name to stats dict with:
            - points, goals_for, goals_against, wins, draws, losses, disciplinary
        """
        matches = self.get_group_matches(group)
        standings = {team: {
            'points': 0,
            'goals_for': 0,
            'goals_against': 0,
            'wins': 0,
            'draws': 0,
            'losses': 0,
            'disciplinary': 0
        } for team in all_teams}

        for match in matches:
            # Update team1
            standings[match.team1]['points'] += match.get_points(match.team1)
            standings[match.team1]['goals_for'] += match.get_goals_for(match.team1)
            standings[match.team1]['goals_against'] += match.get_goals_against(match.team1)
            standings[match.team1]['disciplinary'] += match.get_disciplinary_points(match.team1)

            if match.is_draw:
                standings[match.team1]['draws'] += 1
            elif match.winner == match.team1:
                standings[match.team1]['wins'] += 1
            else:
                standings[match.team1]['losses'] += 1

            # Update team2
            standings[match.team2]['points'] += match.get_points(match.team2)
            standings[match.team2]['goals_for'] += match.get_goals_for(match.team2)
            standings[match.team2]['goals_against'] += match.get_goals_against(match.team2)
            standings[match.team2]['disciplinary'] += match.get_disciplinary_points(match.team2)

            if match.is_draw:
                standings[match.team2]['draws'] += 1
            elif match.winner == match.team2:
                standings[match.team2]['wins'] += 1
            else:
                standings[match.team2]['losses'] += 1

        return standings

    def get_eliminated_teams(self) -> Set[str]:
        """Get all teams that have been eliminated so far."""
        eliminated = set()

        # Teams eliminated in group stage (4th place in completed groups)
        for group in self._by_group:
            if self.has_group_completed(group):
                # Would need to rank teams and identify 4th place
                # This is simplified - proper implementation would rank by FIFA rules
                pass

        # Teams eliminated in knockout stages
        for stage in ['R32', 'R16', 'QF', 'SF', 'Final']:
            for match in self.get_stage_matches(stage):
                if not match.is_draw:  # Knockout must have winner
                    loser = match.team2 if match.winner == match.team1 else match.team1
                    eliminated.add(loser)

        return eliminated


def load_completed_matches(filepath: str = 'data/completed_matches.csv') -> CompletedMatches:
    """
    Load completed matches from CSV file.

    Args:
        filepath: Path to completed matches CSV

    Returns:
        CompletedMatches container
    """
    matches = []

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                match = MatchResult(
                    date=row['date'],
                    stage=row['stage'],
                    group=row['group'] if row['group'] else None,
                    team1=row['team1'],
                    team2=row['team2'],
                    team1_goals=int(row['team1_goals']),
                    team2_goals=int(row['team2_goals']),
                    team1_yellows=int(row['team1_yellows']),
                    team2_yellows=int(row['team2_yellows']),
                    team1_reds=int(row['team1_reds']),
                    team2_reds=int(row['team2_reds']),
                    extra_time=row['extra_time'].lower() == 'true',
                    penalties=row['penalties'].lower() == 'true',
                    winner=row['winner'] if row['winner'] else None
                )
                matches.append(match)
    except FileNotFoundError:
        # No completed matches file yet - that's okay
        pass

    return CompletedMatches(matches)


def get_remaining_group_matches(
    group: str,
    all_teams: List[str],
    completed: CompletedMatches
) -> List[Tuple[str, str]]:
    """
    Get list of remaining matches to simulate in a group.

    Args:
        group: Group letter
        all_teams: All 4 teams in the group
        completed: Completed matches container

    Returns:
        List of (team1, team2) tuples for matches to simulate
    """
    completed_matches = completed.get_group_matches(group)

    # Generate all possible matches
    all_matches = []
    for i in range(len(all_teams)):
        for j in range(i + 1, len(all_teams)):
            all_matches.append((all_teams[i], all_teams[j]))

    # Remove completed matches
    completed_pairs = set()
    for match in completed_matches:
        # Match could be stored as (team1, team2) or (team2, team1)
        completed_pairs.add((match.team1, match.team2))
        completed_pairs.add((match.team2, match.team1))

    remaining = [
        (t1, t2) for t1, t2 in all_matches
        if (t1, t2) not in completed_pairs
    ]

    return remaining
