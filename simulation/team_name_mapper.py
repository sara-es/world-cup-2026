"""Map team names between different sources (bingo cards vs tournament data)."""

from typing import List

# Mapping from bingo card names to tournament data names
BINGO_TO_TOURNAMENT = {
    'United States': 'USA',
    'Türkiye': 'Turkey',
    'Turkey': 'Turkey',  # In case it's already normalized
    'Côte d\'Ivoire': 'Ivory Coast',
    'Ivory Coast': 'Ivory Coast',  # In case it's already normalized
    'Cape Verde': 'Cabo Verde',
    'Cabo Verde': 'Cabo Verde',  # In case it's already normalized
    'Curacao': 'Curaçao',  # Handle ASCII version
    'Curaçao': 'Curaçao',
}


def normalize_team_name(team: str) -> str:
    """
    Normalize a team name to match tournament data.

    Args:
        team: Team name from bingo card or other source

    Returns:
        Normalized team name matching tournament data
    """
    # First, try exact match in mapping
    if team in BINGO_TO_TOURNAMENT:
        return BINGO_TO_TOURNAMENT[team]

    # Return as-is if no mapping found
    return team


def normalize_card_teams(teams: List[str]) -> List[str]:
    """
    Normalize all team names on a card.

    Args:
        teams: List of team names from bingo card

    Returns:
        List of normalized team names
    """
    return [normalize_team_name(team) for team in teams]
