"""R32 bracket construction using Annex C lookup table."""
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, cast

# Add data directory to path to import annexe_c
data_dir = Path(__file__).parent.parent / 'data'
sys.path.insert(0, str(data_dir))
import annexe_c


class R32Bracket:
    """
    Represents the Round of 32 bracket with 16 matches.

    The bracket is determined by:
    - 12 group winners (1A-1L)
    - 12 group runners-up (2A-2L)
    - 8 qualifying third-place teams (assigned via Annex C)
    """

    def __init__(
        self,
        winners: Dict[str, str],
        runners_up: Dict[str, str],
        qualifying_groups: List[str],
        qualifying_teams: List[str]
    ):
        """
        Initialize R32 bracket.

        Args:
            winners: Dict mapping group letter to winner team name
            runners_up: Dict mapping group letter to runner-up team name
            qualifying_groups: List of 8 group letters whose 3rd place teams qualify
            qualifying_teams: List of 8 team names (corresponding to qualifying_groups)
        """
        self.winners = winners
        self.runners_up = runners_up
        self.qualifying_groups = qualifying_groups
        self.qualifying_teams = qualifying_teams

        # Build the bracket using Annex C
        self.matchups = self._build_bracket()

    def _build_bracket(self) -> List[Tuple[str, str]]:
        """
        Build the complete R32 bracket using Annex C lookup and FIFA bracket structure.

        Returns:
            List of 16 (team1, team2) matchup tuples in R32 match order
        """
        # Look up Annex C allocation
        allocation = annexe_c.lookup(self.qualifying_groups)

        # Create mapping from third-place code (e.g., '3E') to team name
        third_place_map = {}
        for group, team in zip(self.qualifying_groups, self.qualifying_teams):
            code = f'3{group}'
            third_place_map[code] = team

        # R32 match slots to bracket positions (from FIFA regulations)
        SLOT_TO_R32_MATCH = {
            '1A': 7,   # Winner A vs 3rd place (Annex C) → R32 match 7
            '1B': 13,  # Winner B vs 3rd place (Annex C) → R32 match 13
            '1D': 9,   # Winner D vs 3rd place (Annex C) → R32 match 9
            '1E': 3,   # Winner E vs 3rd place (Annex C) → R32 match 3
            '1G': 10,  # Winner G vs 3rd place (Annex C) → R32 match 10
            '1I': 4,   # Winner I vs 3rd place (Annex C) → R32 match 4
            '1K': 15,  # Winner K vs 3rd place (Annex C) → R32 match 15
            '1L': 8,   # Winner L vs 3rd place (Annex C) → R32 match 8
        }

        # Initialize 16 R32 matches (indexed 1-16, but we'll use 0-15)
        matchups: List[Optional[Tuple[str, str]]] = [None] * 16

        # Fill in the 8 Winner vs Third-place matches (determined by Annex C)
        for slot in annexe_c.SLOTS:
            winner_group = slot[1]  # e.g., '1A' -> 'A'
            third_code = allocation[slot]  # e.g., '3E'

            winner_team = self.winners[winner_group]
            third_team = third_place_map[third_code]

            match_num = SLOT_TO_R32_MATCH[slot]
            matchups[match_num - 1] = (winner_team, third_team)  # Convert to 0-indexed

        # Fill in the 4 Winner vs Runner-up matches (fixed per FIFA regulations)
        matchups[2 - 1] = (self.winners['F'], self.runners_up['C'])   # R32-2: 1F vs 2C
        matchups[5 - 1] = (self.winners['C'], self.runners_up['F'])   # R32-5: 1C vs 2F
        matchups[12 - 1] = (self.winners['H'], self.runners_up['J'])  # R32-12: 1H vs 2J
        matchups[14 - 1] = (self.winners['J'], self.runners_up['H'])  # R32-14: 1J vs 2H

        # Fill in the 4 Runner-up vs Runner-up matches (fixed per FIFA regulations)
        matchups[1 - 1] = (self.runners_up['A'], self.runners_up['B'])   # R32-1: 2A vs 2B
        matchups[6 - 1] = (self.runners_up['E'], self.runners_up['I'])   # R32-6: 2E vs 2I
        matchups[11 - 1] = (self.runners_up['K'], self.runners_up['L'])  # R32-11: 2K vs 2L
        matchups[16 - 1] = (self.runners_up['D'], self.runners_up['G'])  # R32-16: 2D vs 2G

        # Verify all matches are filled
        assert all(m is not None for m in matchups), "Not all R32 matches were assigned"

        # After assertion, all elements are guaranteed to be tuples
        return cast(List[Tuple[str, str]], matchups)

    def get_matchups(self) -> List[Tuple[str, str]]:
        """Get all R32 matchups."""
        return self.matchups


def build_r32_bracket(
    group_standings: Dict[str, List],
    qualifying_groups: List[str],
    qualifying_teams: List[str]
) -> R32Bracket:
    """
    Build R32 bracket from group stage results and third-place qualifiers.

    Args:
        group_standings: Dict mapping group letter to list of TeamRecords
        qualifying_groups: List of 8 group letters whose 3rd place teams qualify
        qualifying_teams: List of 8 team names that qualify

    Returns:
        R32Bracket instance
    """
    winners = {group: teams[0].team for group, teams in group_standings.items()}
    runners_up = {group: teams[1].team for group, teams in group_standings.items()}

    return R32Bracket(winners, runners_up, qualifying_groups, qualifying_teams)
