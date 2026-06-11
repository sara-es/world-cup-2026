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
from simulation.match_loader import load_completed_matches, CompletedMatches


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

        # Load completed matches
        self.completed_matches = load_completed_matches(matches_path)

        # Simulation parameters
        self.mu = mu
        self.alpha = alpha
        self.k = k
        self.rng = rng if rng is not None else np.random.default_rng()

    def simulate_single_tournament(self) -> Dict[str, List[str]]:
        """
        Simulate one complete tournament.

        Returns:
            Dict mapping stage name to list of teams eliminated in that stage
        """
        # 1. Simulate group stage (using real results where available)
        group_standings = simulate_all_groups(
            self.groups,
            self.strengths,
            self.fifa_ratings,
            self.mu,
            self.alpha,
            self.rng,
            self.completed_matches
        )

        group_result = GroupStageResult(group_standings)

        # Get teams eliminated in group stage (4th place)
        group_eliminated = group_result.get_eliminated()

        # 2. Select best 8 third-place teams
        qualifying_groups, qualifying_teams, third_eliminated = \
            get_third_place_qualifiers(group_standings, self.fifa_ratings)

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

        # Combine all eliminations by stage
        eliminations = {
            'Group': group_eliminated + third_eliminated,
            'R32': knockout_eliminated['R32'],
            'R16': knockout_eliminated['R16'],
            'QF': knockout_eliminated['QF'],
            'SF': knockout_eliminated['SF'],
            'Final': knockout_eliminated['Final']
        }

        return eliminations

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
            eliminations = self.simulate_single_tournament()

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

            # Add to aggregator
            aggregator.add_simulation_result(completion_stages)

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
