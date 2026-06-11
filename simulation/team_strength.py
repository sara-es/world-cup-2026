"""Calculate normalized and blended team strengths."""
from typing import Dict, List


def normalize_ratings(ratings: Dict[str, float]) -> Dict[str, float]:
    """
    Normalize ratings to [0, 1] scale using min-max normalization.

    Args:
        ratings: Dict mapping team name to rating

    Returns:
        Dict mapping team name to normalized rating [0, 1]
    """
    if not ratings:
        return {}

    min_rating = min(ratings.values())
    max_rating = max(ratings.values())

    if max_rating == min_rating:
        # All ratings are the same
        return {team: 0.5 for team in ratings}

    normalized = {}
    for team, rating in ratings.items():
        normalized[team] = (rating - min_rating) / (max_rating - min_rating)

    return normalized


def blend_strengths(
    elo_ratings: Dict[str, float],
    fifa_ratings: Dict[str, float],
    teams: List[str],
    elo_weight: float = 0.6,
    fifa_weight: float = 0.4
) -> Dict[str, float]:
    """
    Calculate blended team strengths from ELO and FIFA ratings.

    Args:
        elo_ratings: Raw ELO ratings
        fifa_ratings: Raw FIFA ratings
        teams: List of all teams to include
        elo_weight: Weight for ELO rating (default 0.6)
        fifa_weight: Weight for FIFA rating (default 0.4)

    Returns:
        Dict mapping team name to blended strength [0, 1]
    """
    # First, ensure we have ratings for all teams
    # If a team is missing from one source, use average
    elo_teams = set(elo_ratings.keys())
    fifa_teams = set(fifa_ratings.keys())
    all_teams = set(teams)

    # Fill in missing teams with average ratings
    avg_elo = sum(elo_ratings.values()) / len(elo_ratings) if elo_ratings else 1500
    avg_fifa = sum(fifa_ratings.values()) / len(fifa_ratings) if fifa_ratings else 1500

    complete_elo = {}
    complete_fifa = {}

    for team in all_teams:
        complete_elo[team] = elo_ratings.get(team, avg_elo)
        complete_fifa[team] = fifa_ratings.get(team, avg_fifa)

    # Normalize both rating systems
    norm_elo = normalize_ratings(complete_elo)
    norm_fifa = normalize_ratings(complete_fifa)

    # Blend the normalized ratings
    blended = {}
    for team in all_teams:
        blended[team] = (
            elo_weight * norm_elo[team] +
            fifa_weight * norm_fifa[team]
        )

    return blended


def get_team_strengths(
    elo_ratings: Dict[str, float],
    fifa_ratings: Dict[str, float],
    teams: List[str]
) -> Dict[str, float]:
    """
    Convenience function to get blended team strengths with default weights.

    Args:
        elo_ratings: Raw ELO ratings
        fifa_ratings: Raw FIFA ratings
        teams: List of all teams

    Returns:
        Dict mapping team name to blended strength [0, 1]
    """
    return blend_strengths(elo_ratings, fifa_ratings, teams)
