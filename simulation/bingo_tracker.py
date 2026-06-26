"""Track bingo card completion through tournament stages."""
from typing import List, Set, Dict, Optional, Any
from collections import defaultdict


class BingoCard:
    """
    Represents a single bingo card with 18 teams.

    Tracks which teams are still alive and at which stage the card completes
    (all 18 teams eliminated).
    """

    def __init__(self, teams: List[str], card_id: Optional[str] = None):
        """
        Initialize bingo card.

        Args:
            teams: List of 18 team names on this card
            card_id: Optional identifier for this card
        """
        if len(teams) != 18:
            raise ValueError(f"Bingo card must have exactly 18 teams, got {len(teams)}")

        self.teams = set(teams)
        self.card_id = card_id
        self.completion_stage = None
        self.teams_alive = set(teams)

    def update(self, eliminated_teams: Set[str]) -> None:
        """
        Update card state based on newly eliminated teams.

        Args:
            eliminated_teams: Set of team names eliminated in this stage
        """
        self.teams_alive -= eliminated_teams

    def is_complete(self) -> bool:
        """Check if all teams on card are eliminated."""
        return len(self.teams_alive) == 0

    def get_num_alive(self) -> int:
        """Get number of teams still alive on this card."""
        return len(self.teams_alive)


class BingoTracker:
    """
    Track multiple bingo cards through a tournament simulation.

    Records at which stage each card completes (all teams eliminated).
    """

    # Tournament stages in order
    STAGES = ['Group', 'R32', 'R16', 'QF', 'SF', 'Final', 'Never']

    def __init__(self, cards: List[List[str]]):
        """
        Initialize tracker with bingo cards.

        Args:
            cards: List of bingo cards, each being a list of 18 team names
        """
        self.cards = [
            BingoCard(teams, card_id=f"Card{i+1}")
            for i, teams in enumerate(cards)
        ]

    def process_stage(
        self,
        stage: str,
        eliminated_teams: List[str]
    ) -> List[str]:
        """
        Process a tournament stage and update card states.

        Args:
            stage: Stage name ('Group', 'R32', etc.)
            eliminated_teams: List of teams eliminated in this stage

        Returns:
            List of card IDs that completed in this stage
        """
        eliminated_set = set(eliminated_teams)
        completed_cards = []

        for card in self.cards:
            # Skip cards already completed
            if card.completion_stage is not None:
                continue

            # Update card with eliminated teams
            card.update(eliminated_set)

            # Check if card is now complete
            if card.is_complete():
                card.completion_stage = stage
                completed_cards.append(card.card_id)

        return completed_cards

    def get_completion_stages(self) -> Dict[str, str]:
        """
        Get completion stage for each card.

        Returns:
            Dict mapping card ID to completion stage
        """
        completion = {}
        for card in self.cards:
            stage = card.completion_stage if card.completion_stage else 'Never'
            completion[card.card_id] = stage
        return completion

    def get_stage_probabilities(self, num_simulations: int) -> Dict[str, Dict[str, float]]:
        """
        Calculate probability of each card being active after each stage.

        Args:
            num_simulations: Total number of simulations run

        Returns:
            Dict mapping card ID to dict of {stage: probability_active_after}
        """
        # This should be called after accumulating results from multiple simulations
        # For single simulation, just return binary completion status
        probabilities = {}

        for card in self.cards:
            card_probs = {}
            completion_stage = card.completion_stage if card.completion_stage else 'Never'

            # Card is active after a stage if it completes AFTER that stage
            stage_index = self.STAGES.index(completion_stage)

            for i, stage in enumerate(self.STAGES[:-1]):  # Exclude 'Never'
                # Active after this stage if completion happens later
                if i < stage_index:
                    card_probs[stage] = 1.0
                else:
                    card_probs[stage] = 0.0

            probabilities[card.card_id] = card_probs

        return probabilities


class BingoSimulationAggregator:
    """
    Aggregate results from multiple bingo simulations.

    Tracks how many times each card completes at each stage across N simulations.
    Also tracks team-level survival probabilities.
    """

    def __init__(self, num_cards: int):
        """
        Initialize aggregator.

        Args:
            num_cards: Number of bingo cards being tracked
        """
        self.num_cards = num_cards
        self.num_simulations = 0

        # completion_counts[card_idx][stage] = count
        self.completion_counts = [
            defaultdict(int) for _ in range(num_cards)
        ]

        # team_survival_counts[card_idx][team][stage] = count of times team survived past stage
        self.team_survival_counts = [
            defaultdict(lambda: defaultdict(int)) for _ in range(num_cards)
        ]

        # completion_match_indices[card_idx] = list of match indices when card completed
        self.completion_match_indices = [
            [] for _ in range(num_cards)
        ]

        # card_active_counts[card_idx][stage] = count of times card was active after stage
        self.card_active_counts = [
            defaultdict(int) for _ in range(num_cards)
        ]

    def add_simulation_result(
        self,
        completion_stages: List[str],
        completion_match_idx: Optional[List[Optional[int]]] = None,
        eliminations: Optional[Dict[str, List[str]]] = None
    ) -> None:
        """
        Add results from one simulation.

        Args:
            completion_stages: List of completion stages, one per card
            completion_match_idx: List of match indices when each card completed (None if never)
            eliminations: Dict mapping stage to list of eliminated teams
        """
        self.num_simulations += 1

        for card_idx, stage in enumerate(completion_stages):
            self.completion_counts[card_idx][stage] += 1

            # Record completion match index
            if completion_match_idx and completion_match_idx[card_idx] is not None:
                self.completion_match_indices[card_idx].append(completion_match_idx[card_idx])

            # Track card active status (same logic as team survival)
            # Card is active after stage S if it completes at a later stage
            if stage == 'Never':
                completion_index = len(BingoTracker.STAGES) - 1
            else:
                completion_index = BingoTracker.STAGES.index(stage)

            # Card is active after a stage if it completes AFTER that stage
            for stage_idx, stage_name in enumerate(BingoTracker.STAGES[:-1]):
                if stage_idx < completion_index:
                    self.card_active_counts[card_idx][stage_name] += 1

    def add_team_survival_data(
        self,
        card_idx: int,
        teams: List[str],
        eliminations: Dict[str, List[str]]
    ) -> None:
        """
        Add team survival data for a single card from one simulation.

        Args:
            card_idx: Index of the card
            teams: List of teams on this card
            eliminations: Dict mapping stage to list of eliminated teams
        """
        # Build set of when each team was eliminated
        team_elimination_stage = {}
        for stage in BingoTracker.STAGES[:-1]:  # Exclude 'Never'
            if stage in eliminations:
                for team in eliminations[stage]:
                    if team in teams:
                        team_elimination_stage[team] = stage

        # For each team, count survival past each stage
        for team in teams:
            # Ensure team has an entry even if never incremented (e.g. eliminated at Group stage)
            _ = self.team_survival_counts[card_idx][team]
            elimination_stage = team_elimination_stage.get(team)

            if elimination_stage is None:
                # Team never eliminated (won tournament)
                elimination_index = len(BingoTracker.STAGES) - 1
            else:
                elimination_index = BingoTracker.STAGES.index(elimination_stage)

            # Team survives past a stage if eliminated AFTER that stage
            for stage_idx, stage in enumerate(BingoTracker.STAGES[:-1]):
                if stage_idx < elimination_index:
                    # Team survived past this stage
                    self.team_survival_counts[card_idx][team][stage] += 1

    def get_probabilities(self) -> List[Dict[str, float]]:
        """
        Calculate probability of each card being active after each stage.
        Uses the same logic as team survival probabilities.

        Returns:
            List of dicts (one per card) mapping stage to P(active after stage)
        """
        probabilities = []

        for card_idx in range(self.num_cards):
            card_probs = {}

            # For each stage, get probability card was active after that stage
            for stage in BingoTracker.STAGES[:-1]:  # Exclude 'Never'
                count = self.card_active_counts[card_idx][stage]
                card_probs[stage] = count / self.num_simulations

            probabilities.append(card_probs)

        return probabilities

    def get_completion_distribution(self) -> List[Dict[str, float]]:
        """
        Get distribution of completion stages for each card.

        Returns:
            List of dicts (one per card) mapping stage to probability
        """
        distributions = []

        for card_idx in range(self.num_cards):
            dist = {}
            for stage in BingoTracker.STAGES:
                count = self.completion_counts[card_idx][stage]
                dist[stage] = count / self.num_simulations

            distributions.append(dist)

        return distributions

    def get_team_survival_probabilities(self) -> List[Dict[str, Dict[str, float]]]:
        """
        Get probability of each team surviving past each stage for each card.

        Returns:
            List of dicts (one per card) where each dict maps team name to
            dict of {stage: probability_survived_past_stage}
        """
        probabilities = []

        for card_idx in range(self.num_cards):
            card_team_probs = {}

            for team, stage_counts in self.team_survival_counts[card_idx].items():
                team_probs = {}
                for stage in BingoTracker.STAGES[:-1]:  # Exclude 'Never'
                    count = stage_counts.get(stage, 0)
                    team_probs[stage] = count / self.num_simulations
                card_team_probs[team] = team_probs

            probabilities.append(card_team_probs)

        return probabilities

    def get_completion_time_stats(self) -> List[Dict[str, Any]]:
        """
        Get completion time statistics for each card.

        Returns:
            List of dicts with completion time stats per card:
            - mean_completion_match: Average match index of completion
            - finishes_first_pct: Percentage of times this card finished first
            - completion_times: List of all completion match indices
        """
        stats = []

        # For each simulation, find which card(s) finished first
        finish_first_counts = [0] * self.num_cards

        # Group completion match indices by simulation
        for sim_idx in range(self.num_simulations):
            # Get completion match index for each card in this simulation
            sim_completions = []
            for card_idx in range(self.num_cards):
                if sim_idx < len(self.completion_match_indices[card_idx]):
                    match_idx = self.completion_match_indices[card_idx][sim_idx]
                    sim_completions.append((match_idx, card_idx))
                else:
                    # Card never completed in this simulation
                    sim_completions.append((float('inf'), card_idx))

            # Find minimum completion time (all tied cards share the prize)
            if sim_completions:
                min_match_idx = min(c[0] for c in sim_completions)
                if min_match_idx != float('inf'):
                    # Count all cards that finished at this time (ties share the prize)
                    for match_idx, card_idx in sim_completions:
                        if match_idx == min_match_idx:
                            finish_first_counts[card_idx] += 1

        # Compute stats for each card
        for card_idx in range(self.num_cards):
            completion_times = self.completion_match_indices[card_idx]

            if completion_times:
                mean_match = sum(completion_times) / len(completion_times)
            else:
                mean_match = float('inf')

            stats.append({
                'mean_completion_match': mean_match,
                'finishes_first_count': finish_first_counts[card_idx],
                'finishes_first_pct': finish_first_counts[card_idx] / self.num_simulations,
                'num_completions': len(completion_times),
                'completion_times': completion_times
            })

        return stats

    def get_first_completion_distribution(self) -> Dict[str, float]:
        """
        Get distribution of when the FIRST card completes across all simulations.

        Returns:
            Dict mapping stage to probability that first completion happens in that stage
        """
        # Map match index to stage
        def match_idx_to_stage(match_idx: int) -> str:
            if match_idx <= 71:
                return 'Group'
            elif match_idx <= 87:
                return 'R32'
            elif match_idx <= 95:
                return 'R16'
            elif match_idx <= 99:
                return 'QF'
            elif match_idx <= 101:
                return 'SF'
            elif match_idx == 102:
                return 'Final'
            else:
                return 'Never'

        stage_counts = {stage: 0 for stage in BingoTracker.STAGES}

        # For each simulation, find the minimum completion time
        for sim_idx in range(self.num_simulations):
            min_match_idx = float('inf')

            # Find minimum across all cards in this simulation
            for card_idx in range(self.num_cards):
                if sim_idx < len(self.completion_match_indices[card_idx]):
                    match_idx = self.completion_match_indices[card_idx][sim_idx]
                    if match_idx < min_match_idx:
                        min_match_idx = match_idx

            # Map to stage and count
            if min_match_idx == float('inf'):
                stage_counts['Never'] += 1
            else:
                stage = match_idx_to_stage(min_match_idx)
                stage_counts[stage] += 1

        # Convert to probabilities
        distribution = {
            stage: count / self.num_simulations
            for stage, count in stage_counts.items()
        }

        return distribution
