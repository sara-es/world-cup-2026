"""Third-place team ranking and selection logic."""
from typing import Dict, List, Tuple
from simulation.group_stage import TeamRecord


def rank_third_place_teams(
    third_place_records: Dict[str, TeamRecord],
    fifa_rankings: Dict[str, float]
) -> List[Tuple[str, TeamRecord]]:
    """
    Rank all third-place teams according to FIFA rules.

    Tiebreaker order:
    1. Points
    2. Goal difference
    3. Goals scored
    4. FIFA ranking (higher rating = better)

    Args:
        third_place_records: Dict mapping group letter to TeamRecord
        fifa_rankings: Dict mapping team name to FIFA ranking

    Returns:
        List of (group_letter, TeamRecord) tuples sorted by ranking
    """
    # Convert to list of (group, record) tuples
    teams_with_groups = list(third_place_records.items())

    # Sort by tiebreaker rules
    ranked = sorted(
        teams_with_groups,
        key=lambda item: (
            item[1].points,
            item[1].goal_diff,
            item[1].goals_for,
            fifa_rankings.get(item[1].team, 0)
        ),
        reverse=True
    )

    return ranked


def select_best_third_place(
    third_place_records: Dict[str, TeamRecord],
    fifa_rankings: Dict[str, float]
) -> Tuple[List[str], List[str]]:
    """
    Select the best 8 third-place teams from 12 groups.

    Args:
        third_place_records: Dict mapping group letter to TeamRecord
        fifa_rankings: Dict mapping team name to FIFA ranking

    Returns:
        Tuple of (qualifying_groups, qualifying_teams)
        - qualifying_groups: List of 8 group letters whose 3rd place teams qualify
        - qualifying_teams: List of 8 team names that qualify
    """
    # Rank all third-place teams
    ranked = rank_third_place_teams(third_place_records, fifa_rankings)

    # Take top 8
    top_8 = ranked[:8]

    qualifying_groups = [group for group, _ in top_8]
    qualifying_teams = [record.team for _, record in top_8]

    return qualifying_groups, qualifying_teams


def get_third_place_qualifiers(
    group_standings: Dict[str, List[TeamRecord]],
    fifa_rankings: Dict[str, float]
) -> Tuple[List[str], List[str], List[str]]:
    """
    Extract and rank third-place teams, returning qualifiers and eliminated.

    Args:
        group_standings: Dict mapping group letter to list of TeamRecords
        fifa_rankings: Dict mapping team name to FIFA ranking

    Returns:
        Tuple of (qualifying_groups, qualifying_teams, eliminated_teams)
    """
    # Extract third-place teams
    third_place = {group: teams[2] for group, teams in group_standings.items()}

    # Select best 8
    qualifying_groups, qualifying_teams = select_best_third_place(
        third_place, fifa_rankings
    )

    # Determine eliminated third-place teams
    all_third_place_teams = [record.team for record in third_place.values()]
    eliminated_teams = [t for t in all_third_place_teams if t not in qualifying_teams]

    return qualifying_groups, qualifying_teams, eliminated_teams
