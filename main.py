"""CLI entry point for World Cup 2026 bingo simulation."""
import argparse
import json
import math
from typing import List
from simulation.simulator import WorldCupSimulator


def load_cards_from_file(filename: str) -> List[List[str]]:
    """
    Load bingo cards from JSON file.

    Expected format: List of lists, each inner list has 18 team names.
    """
    with open(filename, 'r', encoding='utf-8') as f:
        cards = json.load(f)

    # Validate
    for i, card in enumerate(cards):
        if len(card) != 18:
            raise ValueError(f"Card {i+1} has {len(card)} teams, expected 18")

    return cards


def print_results(aggregator, cards):
    """Print simulation results in readable format."""
    print("\n" + "="*80)
    print("WORLD CUP 2026 BINGO SIMULATION RESULTS")
    print("="*80)

    probabilities = aggregator.get_probabilities()
    distributions = aggregator.get_completion_distribution()

    for i, (card, probs, dist) in enumerate(zip(cards, probabilities, distributions)):
        print(f"\nCard {i+1}: {', '.join(card[:5])}... ({len(card)} teams)")
        print("-" * 40)

        print("\nProbability card is still active after each stage:")
        for stage in ['Group', 'R32', 'R16', 'QF', 'SF', 'Final']:
            prob = probs[stage]
            bar = '#' * int(prob * 50)
            print(f"  {stage:10s}: {prob:6.2%} {bar}")

        print("\nCompletion stage distribution:")
        for stage in ['Group', 'R32', 'R16', 'QF', 'SF', 'Final', 'Never']:
            prob = dist[stage]
            if prob > 0:
                bar = '#' * int(prob * 50)
                print(f"  {stage:10s}: {prob:6.2%} {bar}")

    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(
        description='World Cup 2026 Bingo Monte Carlo Simulation'
    )

    parser.add_argument(
        '--cards',
        type=str,
        help='JSON file containing bingo cards (list of 18-team lists)'
    )

    parser.add_argument(
        '--teams',
        type=str,
        nargs='+',
        help='Teams for a single card (provide 18 team names)'
    )

    parser.add_argument(
        '-n', '--num-sims',
        type=int,
        default=10000,
        help='Number of simulations to run (default: 10000)'
    )

    parser.add_argument(
        '--mu',
        type=float,
        default=math.log(1.3),
        help='Poisson base rate parameter (default: ln(1.3))'
    )

    parser.add_argument(
        '--alpha',
        type=float,
        default=0.35,
        help='Poisson strength effect parameter (default: 0.35)'
    )

    parser.add_argument(
        '--k',
        type=float,
        default=0.5,
        help='Bradley-Terry exponent (default: 0.5)'
    )

    parser.add_argument(
        '--grid-search',
        action='store_true',
        help='Run grid search over alpha and k parameters'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print progress updates'
    )

    parser.add_argument(
        '--output',
        type=str,
        help='Save results to JSON file'
    )

    args = parser.parse_args()

    # Load or create cards
    if args.cards:
        cards = load_cards_from_file(args.cards)
    elif args.teams:
        if len(args.teams) != 18:
            print(f"Error: Need exactly 18 teams, got {len(args.teams)}")
            return
        cards = [args.teams]
    else:
        # Default example card
        cards = [[
            'Mexico', 'South Africa', 'Canada', 'Brazil',
            'USA', 'Germany', 'Belgium', 'Spain',
            'France', 'Argentina', 'Portugal', 'England',
            'Morocco', 'Netherlands', 'Japan', 'Senegal',
            'Australia', 'Turkey'
        ]]
        print("No cards specified, using example card")

    # Initialize simulator
    simulator = WorldCupSimulator(
        mu=args.mu,
        alpha=args.alpha,
        k=args.k
    )

    if args.grid_search:
        print("Running grid search...")
        alpha_values = [0.3, 0.5, 0.7]
        k_values = [1.0, 1.5, 2.0]

        results = simulator.grid_search(
            cards,
            alpha_values,
            k_values,
            num_simulations=args.num_sims,
            verbose=args.verbose
        )

        print("\nGrid search results:")
        for (alpha, k), aggregator in results.items():
            print(f"\nalpha={alpha}, k={k}")
            probs = aggregator.get_probabilities()[0]  # First card
            print(f"  P(active after R16) = {probs['R16']:.2%}")

    else:
        print(f"Running {args.num_sims} simulations...")
        print(f"Parameters: mu={args.mu:.3f}, alpha={args.alpha}, k={args.k}")

        aggregator = simulator.simulate_bingo_cards(
            cards,
            num_simulations=args.num_sims,
            verbose=args.verbose
        )

        # Print results
        print_results(aggregator, cards)

        # Save to file if requested
        if args.output:
            results_dict = {
                'num_simulations': args.num_sims,
                'parameters': {
                    'mu': args.mu,
                    'alpha': args.alpha,
                    'k': args.k
                },
                'probabilities': aggregator.get_probabilities(),
                'distributions': aggregator.get_completion_distribution()
            }

            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(results_dict, f, indent=2)

            print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
