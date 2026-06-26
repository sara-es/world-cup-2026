"""Main Monte Carlo simulator orchestrating all components."""
import math
import numpy as np
from typing import Dict, List, Optional
from simulation.data_loader import load_all_data, get_all_teams
from simulation.team_strength import get_team_strengths
from simulation.group_stage import simulate_all_groups, GroupStageResult
from simulation.third_place import get_third_place_qualifiers
from simulation.bracket import build_r32_bracket
from simulation.knockout import KnockoutStage
from simulation.bingo_tracker import BingoTracker, BingoSimulationAggregator
from simulation.match_loader import load_completed_matches, CompletedMatches, load_fair_play


class WorldCupSimulator:
    """
    Monte Carlo simulator for World Cup 2026 bingo.

    Simulates tournament outcomes and tracks bingo card completion.
    """

    def __init__(
        self,
        elo_path: str = 'data/elo_ratings.tsv',
        fifa_path: str = 'data/fifa_ratings.csv',
        groups_path: str = 'data/bracket.csv',
        matches_path: str = 'data/completed_matches.csv',
        fair_play_path: str = 'data/fair_play.csv',
        mu: float = math.log(1.3),
        alpha: float = 0.35,
        k: float = 0.5,
        rng: Optional[np.random.Generator] = None
    ):
        """
        Initialize simulator with data and parameters.

        Args:
            elo_path: Path to ELO ratings file
            fifa_path: Path to FIFA ratings file
            groups_path: Path to groups/bracket file
            matches_path: Path to completed matches CSV (optional)
            mu: Poisson base rate parameter (default: ln(1.3))
            alpha: Poisson strength effect (default: 0.35)
            k: Bradley-Terry exponent (default: 0.5)
            rng: Random number generator (default: np.random.default_rng())
        """
        # Load data
        self.elo_ratings, self.fifa_ratings, self.groups = load_all_data(
            elo_path, fifa_path, groups_path
        )

        # Get all teams
        self.all_teams = get_all_teams(self.groups)

        # Calculate team strengths
        self.strengths = get_team_strengths(
            self.elo_ratings,
            self.fifa_ratings,
            self.all_teams
        )

        # Load completed matches and fair play data
        self.completed_matches = load_completed_matches(matches_path)
        self.fair_play = load_fair_play(fair_play_path)

        # Simulation parameters
        self.mu = mu
        self.alpha = alpha
        self.k = k
        self.rng = rng if rng is not None else np.random.default_rng()

    def simulate_single_tournament(self) -> tuple[Dict[str, List[str]], Dict[str, int]]:
        """
        Simulate one complete tournament.

        Returns:
            Tuple of:
            - Dict mapping stage name to list of teams eliminated in that stage
            - Dict mapping team name to match_index when eliminated
        """
        # Match indices:
        # Group stage: 72 matches (0-71), eliminations happen at match 71
        # R32: 16 matches (72-87)
        # R16: 8 matches (88-95)
        # QF: 4 matches (96-99)
        # SF: 2 matches (100-101)
        # Final: 1 match (102)

        # 1. Simulate group stage (using real results where available)
        group_standings = simulate_all_groups(
            self.groups,
            self.strengths,
            self.fifa_ratings,
            self.mu,
            self.alpha,
            self.rng,
            self.completed_matches,
            fair_play=self.fair_play
        )

        group_result = GroupStageResult(group_standings)

        # Get teams eliminated in group stage (4th place)
        group_eliminated = group_result.get_eliminated()

        # 2. Select best 8 third-place teams
        qualifying_groups, qualifying_teams, third_eliminated = \
            get_third_place_qualifiers(group_standings, self.fifa_ratings)

        # Track elimination match indices
        team_elimination_match = {}

        # Group stage eliminations happen after all group matches (match 71)
        for team in group_eliminated + third_eliminated:
            team_elimination_match[team] = 71

        # 3. Build R32 bracket
        r32_bracket = build_r32_bracket(
            group_standings,
            qualifying_groups,
            qualifying_teams
        )

        # 4. Simulate knockout stage
        knockout = KnockoutStage(
            r32_bracket.get_matchups(),
            self.strengths,
            self.k
        )
        knockout.simulate()

        knockout_eliminated = knockout.get_eliminated_by_stage()

        # Assign match indices for knockout eliminations
        match_idx = 72  # Start of R32
        for stage in ['R32', 'R16', 'QF', 'SF', 'Final']:
            num_matches = {'R32': 16, 'R16': 8, 'QF': 4, 'SF': 2, 'Final': 1}[stage]
            # Eliminated teams are spread across matches in this stage
            teams_in_stage = knockout_eliminated[stage]
            teams_per_match = 2 if stage != 'Final' else 1  # Final only eliminates 1 team

            for i, team in enumerate(teams_in_stage):
                # Distribute teams across matches in this stage
                team_match_offset = i // teams_per_match
                team_elimination_match[team] = match_idx + team_match_offset

            match_idx += num_matches

        # Combine all eliminations by stage
        eliminations = {
            'Group': group_eliminated + third_eliminated,
            'R32': knockout_eliminated['R32'],
            'R16': knockout_eliminated['R16'],
            'QF': knockout_eliminated['QF'],
            'SF': knockout_eliminated['SF'],
            'Final': knockout_eliminated['Final']
        }

        return eliminations, team_elimination_match

    def simulate_bingo_cards(
        self,
        cards: List[List[str]],
        num_simulations: int = 10000,
        verbose: bool = False
    ) -> BingoSimulationAggregator:
        """
        Run Monte Carlo simulation for bingo cards.

        Args:
            cards: List of bingo cards (each is list of 18 team names)
            num_simulations: Number of simulations to run (default: 10000)
            verbose: Print progress updates (default: False)

        Returns:
            BingoSimulationAggregator with accumulated results
        """
        aggregator = BingoSimulationAggregator(len(cards))

        for sim_num in range(num_simulations):
            if verbose and (sim_num + 1) % 1000 == 0:
                print(f"Simulation {sim_num + 1}/{num_simulations}")

            # Run one tournament simulation
            eliminations, team_elimination_match = self.simulate_single_tournament()

            # Track bingo cards through this simulation
            tracker = BingoTracker(cards)

            # Process each stage in order
            for stage in BingoTracker.STAGES[:-1]:  # Exclude 'Never'
                if stage in eliminations:
                    tracker.process_stage(stage, eliminations[stage])

            # Get completion stages for all cards
            completion_stages_dict = tracker.get_completion_stages()

            # Convert to list (in card order)
            completion_stages = [
                completion_stages_dict[f"Card{i+1}"]
                for i in range(len(cards))
            ]

            # Compute completion match index for each card
            # (max elimination match index among the card's 18 teams)
            completion_match_indices = []
            for card_teams in cards:
                card_team_matches = []
                for team in card_teams:
                    if team in team_elimination_match:
                        card_team_matches.append(team_elimination_match[team])

                if card_team_matches:
                    # Card completes when last team is eliminated
                    completion_match_indices.append(max(card_team_matches))
                else:
                    # Card never completes (shouldn't happen)
                    completion_match_indices.append(None)

            # Add to aggregator
            aggregator.add_simulation_result(completion_stages, completion_match_indices)

            # Track team survival for each card
            for card_idx, card_teams in enumerate(cards):
                aggregator.add_team_survival_data(card_idx, card_teams, eliminations)

        return aggregator

    def grid_search(
        self,
        cards: List[List[str]],
        alpha_values: List[float],
        k_values: List[float],
        num_simulations: int = 1000,
        verbose: bool = False
    ) -> Dict[tuple, BingoSimulationAggregator]:
        """
        Run grid search over alpha and k parameters.

        Args:
            cards: List of bingo cards
            alpha_values: List of alpha values to try
            k_values: List of k values to try
            num_simulations: Simulations per parameter combination
            verbose: Print progress updates

        Returns:
            Dict mapping (alpha, k) to BingoSimulationAggregator
        """
        results = {}

        for alpha in alpha_values:
            for k in k_values:
                if verbose:
                    print(f"\nRunning simulations for alpha={alpha}, k={k}")

                # Update parameters
                self.alpha = alpha
                self.k = k

                # Run simulations
                aggregator = self.simulate_bingo_cards(
                    cards,
                    num_simulations,
                    verbose=verbose
                )

                results[(alpha, k)] = aggregator

        return results


def quick_simulation(
    cards: List[List[str]],
    num_simulations: int = 10000,
    mu: float = math.log(1.3),
    alpha: float = 0.35,
    k: float = 0.5,
    matches_path: str = 'data/completed_matches.csv',
    verbose: bool = False
) -> BingoSimulationAggregator:
    """
    Convenience function for quick simulation with default settings.

    Args:
        cards: List of bingo cards (each is list of 18 team names)
        num_simulations: Number of simulations (default: 10000)
        mu: Poisson base rate (default: ln(1.3))
        alpha: Poisson strength effect (default: 0.35)
        k: Bradley-Terry exponent (default: 0.5)
        matches_path: Path to completed matches CSV (default: 'data/completed_matches.csv')
        verbose: Print progress (default: False)

    Returns:
        BingoSimulationAggregator with results
    """
    simulator = WorldCupSimulator(mu=mu, alpha=alpha, k=k, matches_path=matches_path)
    return simulator.simulate_bingo_cards(cards, num_simulations, verbose)
