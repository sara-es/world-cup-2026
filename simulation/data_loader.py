"""Load and preprocess World Cup 2026 data."""
import csv
from pathlib import Path
from typing import Dict, List, Tuple


# Team name normalization mapping
TEAM_NAME_MAPPING = {
    # Handle common variations
    'United States': 'USA',
    'IR Iran': 'Iran',
    'Korea Republic': 'South Korea',
    'Côte d\'Ivoire': 'Ivory Coast',
    'Congo DR': 'DR Congo',
    'Cape Verde': 'Cabo Verde',
    'Türkiye': 'Turkey',
    'Korea DPR': 'North Korea',
    'China PR': 'China',
}

# ELO code to full name mapping (from the files)
ELO_TO_NAME = {
    'ES': 'Spain', 'AR': 'Argentina', 'FR': 'France', 'EN': 'England',
    'BR': 'Brazil', 'PT': 'Portugal', 'CO': 'Colombia', 'NL': 'Netherlands',
    'EC': 'Ecuador', 'DE': 'Germany', 'NO': 'Norway', 'HR': 'Croatia',
    'TR': 'Turkey', 'JP': 'Japan', 'BE': 'Belgium', 'UY': 'Uruguay',
    'CH': 'Switzerland', 'MX': 'Mexico', 'DK': 'Denmark', 'IT': 'Italy',
    'SN': 'Senegal', 'PY': 'Paraguay', 'AT': 'Austria', 'MA': 'Morocco',
    'CA': 'Canada', 'SQ': 'Scotland', 'UA': 'Ukraine', 'AU': 'Australia',
    'IR': 'Iran', 'RU': 'Russia', 'NG': 'Nigeria', 'DZ': 'Algeria',
    'KR': 'South Korea', 'GR': 'Greece', 'CZ': 'Czech Republic', 'RS': 'Serbia',
    'VE': 'Venezuela', 'PA': 'Panama', 'US': 'USA', 'CL': 'Chile',
    'KO': 'Kosovo', 'UZ': 'Uzbekistan', 'SE': 'Sweden', 'HU': 'Hungary',
    'PL': 'Poland', 'PE': 'Peru', 'IE': 'Republic of Ireland', 'EG': 'Egypt',
    'CI': 'Ivory Coast', 'WA': 'Wales', 'SI': 'Slovenia', 'JO': 'Jordan',
    'SK': 'Slovakia', 'GE': 'Georgia', 'CD': 'DR Congo', 'IL': 'Israel',
    'RO': 'Romania', 'BO': 'Bolivia', 'TN': 'Tunisia', 'AL': 'Albania',
    'CM': 'Cameroon', 'CR': 'Costa Rica', 'IQ': 'Iraq', 'EI': 'Republic of Ireland',
    'BA': 'Bosnia and Herzegovina', 'NM': 'Northern Ireland', 'ML': 'Mali',
    'CV': 'Cape Verde', 'SA': 'Saudi Arabia', 'HN': 'Honduras', 'IS': 'Iceland',
    'NZ': 'New Zealand', 'HT': 'Haiti', 'AO': 'Angola', 'AE': 'United Arab Emirates',
    'FI': 'Finland', 'BF': 'Burkina Faso', 'JM': 'Jamaica', 'BY': 'Belarus',
    'ZA': 'South Africa', 'GH': 'Ghana', 'GT': 'Guatemala', 'OM': 'Oman',
    'SY': 'Syria', 'PS': 'Palestine', 'GN': 'Guinea', 'ME': 'Montenegro',
    'BG': 'Bulgaria', 'LU': 'Luxembourg', 'NS': 'North Macedonia', 'CW': 'Curaçao',
    'SR': 'Suriname', 'KZ': 'Kazakhstan', 'CN': 'China', 'KD': 'North Korea',
    'QA': 'Qatar', 'LY': 'Libya', 'GM': 'Gambia', 'BH': 'Bahrain',
    'BJ': 'Benin', 'GA': 'Gabon', 'UG': 'Uganda', 'TT': 'Trinidad and Tobago',
}


def load_elo_ratings(filepath: str = 'elo_ratings.tsv') -> Dict[str, float]:
    """
    Load ELO ratings from TSV file.

    Returns:
        Dict mapping team name to ELO rating
    """
    elo_ratings = {}

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 4:
                continue

            # Format: rank→num  num  CODE  rating  ...
            # Example: 1→1  1  ES  2157  ...
            try:
                code = parts[2]
                rating = float(parts[3])

                if code in ELO_TO_NAME:
                    team_name = ELO_TO_NAME[code]
                    elo_ratings[team_name] = rating
            except (ValueError, IndexError):
                continue

    return elo_ratings


def load_fifa_ratings(filepath: str = 'fifa_ratings.csv') -> Dict[str, float]:
    """
    Load FIFA ratings from CSV file.

    Returns:
        Dict mapping team name to FIFA rating
    """
    fifa_ratings = {}

    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            team_name = row['team'].strip()
            rating = float(row['rating'])

            # Normalize team name
            if team_name in TEAM_NAME_MAPPING:
                team_name = TEAM_NAME_MAPPING[team_name]

            fifa_ratings[team_name] = rating

    return fifa_ratings


def load_groups(filepath: str = 'bracket.csv') -> Dict[str, List[str]]:
    """
    Load group assignments from CSV file.

    Returns:
        Dict mapping group letter to list of 4 team names
    """
    groups = {}

    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            group_letter = row['group'].strip()
            teams = [
                row['team_1'].strip(),
                row['team_2'].strip(),
                row['team_3'].strip(),
                row['team_4'].strip(),
            ]

            # Normalize team names
            teams = [TEAM_NAME_MAPPING.get(t, t) for t in teams]

            groups[group_letter] = teams

    return groups


def get_all_teams(groups: Dict[str, List[str]]) -> List[str]:
    """Get list of all 48 teams from groups."""
    teams = []
    for group_teams in groups.values():
        teams.extend(group_teams)
    return teams


def load_all_data(
    elo_path: str = 'data/elo_ratings.tsv',
    fifa_path: str = 'data/fifa_ratings.csv',
    groups_path: str = 'data/bracket.csv'
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, List[str]]]:
    """
    Load all data files.

    Returns:
        Tuple of (elo_ratings, fifa_ratings, groups)
    """
    elo_ratings = load_elo_ratings(elo_path)
    fifa_ratings = load_fifa_ratings(fifa_path)
    groups = load_groups(groups_path)

    return elo_ratings, fifa_ratings, groups
