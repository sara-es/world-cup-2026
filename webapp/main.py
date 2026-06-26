"""FastAPI web application for World Cup 2026 Bingo simulation."""
import math
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import sys
from pathlib import Path
import uuid
import requests
from bs4 import BeautifulSoup
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from simulation.simulator import WorldCupSimulator
from simulation.data_loader import load_all_data, get_all_teams
from simulation.team_name_mapper import normalize_card_teams
from simulation.group_stage import rank_group, TeamRecord
from simulation.third_place import get_third_place_qualifiers, rank_third_place_teams
from simulation.group_stage import simulate_all_groups
from simulation.bracket import R32Bracket
from simulation.knockout import bradley_terry_win_prob


app = FastAPI(
    title="World Cup 2026 Bingo Simulator",
    description="Monte Carlo simulation for World Cup bingo cards",
    version="1.0.0"
)

# Mount static files
app.mount("/static", StaticFiles(directory="webapp/static"), name="static")

# Global simulator instance (initialized on startup)
simulator: Optional[WorldCupSimulator] = None
all_teams_list: List[str] = []


class BingoCardRequest(BaseModel):
    """Request model for bingo simulation."""
    cards: List[List[str]] = Field(
        ...,
        description="List of bingo cards, each containing 18 team names",
        min_items=1,
        max_items=10
    )
    num_simulations: int = Field(
        default=10000,
        description="Number of Monte Carlo simulations to run",
        ge=100,
        le=100000
    )
    mu: float = Field(
        default=math.log(1.3),
        description="Poisson base rate parameter"
    )
    alpha: float = Field(
        default=0.35,
        description="Poisson strength effect parameter",
        ge=0.0,
        le=15.0
    )
    k: float = Field(
        default=0.5,
        description="Bradley-Terry exponent parameter",
        ge=0.0,
        le=15.0
    )


class BingoCardResponse(BaseModel):
    """Response model for bingo simulation."""
    num_simulations: int
    parameters: Dict[str, float]
    eliminated_teams: List[str] = []
    results: List[Dict[str, Any]]


class TeamsListResponse(BaseModel):
    """Response model for available teams."""
    teams: List[str]


class RankCardsRequest(BaseModel):
    """Request model for ranking all scraped cards."""
    num_simulations: int = Field(
        default=10000,
        description="Number of Monte Carlo simulations to run",
        ge=100,
        le=100000
    )
    mu: float = Field(
        default=math.log(1.3),
        description="Poisson base rate parameter"
    )
    alpha: float = Field(
        default=0.35,
        description="Poisson strength effect parameter",
        ge=0.0,
        le=15.0
    )
    k: float = Field(
        default=0.5,
        description="Bradley-Terry exponent parameter",
        ge=0.0,
        le=15.0
    )


class RankCardsResponse(BaseModel):
    """Response model for ranked cards."""
    num_cards: int
    num_simulations: int
    parameters: Dict[str, float]
    rankings: List[Dict[str, Any]]
    first_completion_distribution: Dict[str, float]


import csv

BINGO_URL = "https://stain.github.io/worldcupbingo/hundreds.html"
NAMESPACE_WORLDCUP = uuid.UUID("dedaeff9-2834-51b1-afda-9d8e2ea53d38")
BINGO_CARDS_CSV = "data/bingo_cards.csv"


def compute_board_id(teams: List[str]) -> str:
    """Replicates getBoardHash() from worldcupbingo.py."""
    return str(uuid.uuid5(NAMESPACE_WORLDCUP, "\n".join(sorted(teams))))


def scrape_bingo_cards() -> List[Dict[str, Any]]:
    """
    Scrape bingo cards from worldcupbingo hundreds.html.

    Returns:
        List of dicts with 'board_id' and 'teams' keys
    """
    print(f"Fetching {BINGO_URL} ...")
    r = requests.get(BINGO_URL, timeout=30)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")
    boards = soup.find_all("div", class_="bingo")
    print(f"Found {len(boards)} boards.")

    if not boards:
        raise ValueError("No boards found – the page structure may have changed.")

    cards = []
    for i, board in enumerate(boards, start=1):
        # Board ID embedded in the page
        h1 = board.find("h1")
        embedded_id = h1["id"].strip() if (h1 and h1.get("id")) else None

        # Fallback: read from the <div class='hash'><b>...</b></div>
        if not embedded_id:
            hash_div = board.find("div", class_="hash")
            b_tag = hash_div.find("b") if hash_div else None
            embedded_id = b_tag.get_text(strip=True) if b_tag else None

        # Teams: text content of each <td> in the board's table
        table = board.find("table")
        if not table:
            print(f"  Board {i}: no table found, skipping.")
            continue
        teams = [td.get_text(strip=True) for td in table.find_all("td")]

        # Cross-check by recomputing the ID from the teams
        computed_id = compute_board_id(teams)
        board_id = embedded_id or computed_id

        cards.append({
            "board_id": board_id,
            "teams": teams
        })

    print(f"Scraped {len(cards)} cards")
    return cards


def save_cards_to_csv(cards: List[Dict[str, Any]], filepath: str = BINGO_CARDS_CSV):
    """
    Save scraped cards to CSV file.

    Args:
        cards: List of card dicts with 'board_id' and 'teams'
        filepath: Path to CSV file
    """
    if not cards:
        return

    # Determine max number of teams (should be 18)
    max_teams = max(len(card["teams"]) for card in cards)
    team_headers = [f"team_{i}" for i in range(1, max_teams + 1)]

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["board_id"] + team_headers)
        for card in cards:
            teams = card["teams"]
            # Pad with empty strings if needed
            row = [card["board_id"]] + teams + [""] * (max_teams - len(teams))
            writer.writerow(row)

    print(f"Saved {len(cards)} cards to {filepath}")


def load_cards_from_csv(filepath: str = BINGO_CARDS_CSV) -> List[Dict[str, Any]]:
    """
    Load cards from CSV file.

    Args:
        filepath: Path to CSV file

    Returns:
        List of dicts with 'board_id' and 'teams' keys
    """
    cards = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        headers = next(reader)  # Skip header row

        for row in reader:
            board_id = row[0]
            # Teams are all non-empty values after board_id
            teams = [team for team in row[1:] if team]
            cards.append({
                "board_id": board_id,
                "teams": teams
            })

    print(f"Loaded {len(cards)} cards from {filepath}")
    return cards


def load_boards_in_play(filepath: str = "data/boards_in_play.csv") -> set:
    """
    Load board IDs that are in play (have been bought).

    Args:
        filepath: Path to boards_in_play CSV file

    Returns:
        Set of board IDs that are in play
    """
    boards_in_play = set()
    with open(filepath, "r", encoding="utf-8-sig") as f:  # utf-8-sig to handle BOM
        reader = csv.DictReader(f)
        for row in reader:
            board_id = row["Board"].strip()
            in_play = row["In Play"].strip().upper()
            if in_play == "TRUE":
                boards_in_play.add(board_id)

    print(f"Loaded {len(boards_in_play)} boards in play from {filepath}")
    return boards_in_play


@app.on_event("startup")
async def startup_event():
    """Initialize simulator on startup."""
    global simulator, all_teams_list

    # Load data and initialize simulator
    elo_ratings, fifa_ratings, groups = load_all_data()
    all_teams_list = sorted(get_all_teams(groups))

    simulator = WorldCupSimulator()

    print("Simulator initialized successfully")
    print(f"Loaded {len(all_teams_list)} teams")


@app.get("/")
async def root():
    """Serve the main HTML page."""
    return FileResponse("webapp/static/index.html")


@app.get("/api/teams", response_model=TeamsListResponse)
async def get_teams():
    """Get list of all available teams."""
    return TeamsListResponse(teams=all_teams_list)


@app.post("/api/simulate", response_model=BingoCardResponse)
async def simulate_cards(request: BingoCardRequest):
    """
    Run Monte Carlo simulation for bingo cards.

    Args:
        request: BingoCardRequest with cards and simulation parameters

    Returns:
        BingoCardResponse with simulation results
    """
    if simulator is None:
        raise HTTPException(status_code=500, detail="Simulator not initialized")

    # Normalize and validate cards
    normalized_cards = []
    for i, card in enumerate(request.cards):
        if len(card) != 18:
            raise HTTPException(
                status_code=400,
                detail=f"Card {i+1} must have exactly 18 teams, got {len(card)}"
            )

        # Normalize team names
        normalized_card = normalize_card_teams(card)
        normalized_cards.append(normalized_card)

        # Check that all teams are valid
        invalid_teams = [t for t in normalized_card if t not in all_teams_list]
        if invalid_teams:
            raise HTTPException(
                status_code=400,
                detail=f"Card {i+1} contains invalid teams: {', '.join(invalid_teams)}"
            )

    # Update simulator parameters
    simulator.mu = request.mu
    simulator.alpha = request.alpha
    simulator.k = request.k

    # Run simulation
    try:
        aggregator = simulator.simulate_bingo_cards(
            normalized_cards,
            num_simulations=request.num_simulations,
            verbose=False
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Simulation failed: {str(e)}"
        )

    # Format results
    probabilities = aggregator.get_probabilities()
    distributions = aggregator.get_completion_distribution()
    team_survival = aggregator.get_team_survival_probabilities()

    results = []
    for i, (card, probs, dist, team_probs) in enumerate(
        zip(request.cards, probabilities, distributions, team_survival)
    ):
        results.append({
            'card_id': i + 1,
            'teams': card,
            'active_probabilities': probs,
            'completion_distribution': dist,
            'team_survival_probabilities': team_probs
        })

    eliminated_teams = list(simulator.completed_matches.get_eliminated_teams())

    return BingoCardResponse(
        num_simulations=request.num_simulations,
        parameters={
            'mu': request.mu,
            'alpha': request.alpha,
            'k': request.k
        },
        eliminated_teams=eliminated_teams,
        results=results
    )


@app.post("/api/scrape-cards")
async def scrape_and_save_cards():
    """
    Scrape cards from worldcupbingo and save to CSV.

    Returns:
        Status message with number of cards scraped
    """
    try:
        cards = scrape_bingo_cards()
        save_cards_to_csv(cards)
        return {
            "status": "success",
            "num_cards": len(cards),
            "message": f"Scraped and saved {len(cards)} cards to {BINGO_CARDS_CSV}"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to scrape cards: {str(e)}"
        )


@app.post("/api/rank-cards", response_model=RankCardsResponse)
async def rank_all_cards(request: RankCardsRequest):
    """
    Rank all bingo cards by completion probability.
    Loads cards from CSV file (generated by scraping).

    Args:
        request: RankCardsRequest with simulation parameters

    Returns:
        RankCardsResponse with ranked cards
    """
    if simulator is None:
        raise HTTPException(status_code=500, detail="Simulator not initialized")

    # Load cards from CSV
    try:
        scraped_cards = load_cards_from_csv()
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Cards file not found. Please scrape cards first using the 'Update Cards from Web' button."
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load cards: {str(e)}"
        )

    if not scraped_cards:
        raise HTTPException(status_code=404, detail="No cards found")

    # Load boards that are in play (bought)
    try:
        boards_in_play = load_boards_in_play()
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"boards_in_play.csv file not found."
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load boards_in_play.csv: {str(e)}"
        )

    # Filter to only cards that are in play
    scraped_cards = [card for card in scraped_cards if card["board_id"] in boards_in_play]

    if not scraped_cards:
        raise HTTPException(status_code=404, detail="No boards in play found")

    # Extract and normalize team lists for simulation
    cards = [card["teams"] for card in scraped_cards]
    normalized_cards = [normalize_card_teams(card) for card in cards]

    # Validate cards
    for i, card in enumerate(normalized_cards):
        if len(card) != 18:
            raise HTTPException(
                status_code=400,
                detail=f"Card {i+1} (ID: {scraped_cards[i]['board_id']}) must have exactly 18 teams, got {len(card)}"
            )

        # Check that all teams are valid
        invalid_teams = [t for t in card if t not in all_teams_list]
        if invalid_teams:
            raise HTTPException(
                status_code=400,
                detail=f"Card {i+1} (ID: {scraped_cards[i]['board_id']}) contains invalid teams: {', '.join(invalid_teams)}"
            )

    # Update simulator parameters
    simulator.mu = request.mu
    simulator.alpha = request.alpha
    simulator.k = request.k

    # Run simulation for all cards
    try:
        aggregator = simulator.simulate_bingo_cards(
            normalized_cards,
            num_simulations=request.num_simulations,
            verbose=True
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Simulation failed: {str(e)}"
        )

    # Get completion distributions, active probabilities, and time statistics
    distributions = aggregator.get_completion_distribution()
    active_probabilities = aggregator.get_probabilities()
    completion_time_stats = aggregator.get_completion_time_stats()
    first_completion_dist = aggregator.get_first_completion_distribution()

    # Calculate rankings for each stage
    eliminated_teams = simulator.completed_matches.get_eliminated_teams()
    rankings = []

    for i, (card_data, dist, active_probs, time_stats) in enumerate(
        zip(scraped_cards, distributions, active_probabilities, completion_time_stats)
    ):
        n_eliminated = sum(1 for t in normalized_cards[i] if t in eliminated_teams)
        rankings.append({
            'board_id': card_data['board_id'],
            'teams': card_data['teams'],
            'mean_completion_match': time_stats['mean_completion_match'],
            'finishes_first_pct': time_stats['finishes_first_pct'],
            'finishes_first_count': time_stats['finishes_first_count'],
            'num_completions': time_stats['num_completions'],
            'completion_distribution': dist,
            'active_probabilities': active_probs,
            'prob_never_completes': dist['Never'],
            'n_teams_eliminated': n_eliminated,
        })

    # Sort by number of teams eliminated (higher = worse = more teams out), then by finishes_first_pct descending
    rankings.sort(key=lambda x: (-x['n_teams_eliminated'], -x['finishes_first_pct']))

    return RankCardsResponse(
        num_cards=len(rankings),
        num_simulations=request.num_simulations,
        parameters={
            'mu': request.mu,
            'alpha': request.alpha,
            'k': request.k
        },
        rankings=rankings,
        first_completion_distribution=first_completion_dist
    )


def _compute_confirmed_positions(standings_dict: Dict, ranked: List,
                                  remaining_matches: List) -> Dict:
    """
    Points-based confirmation of group positions.
    Accounts for mutual-exclusivity when two threats play each other (not involving the leader).
    Conservative: ignores GD tiebreakers, so may under-report confirmed positions.
    """
    max_pts = {
        t: s['points'] + 3 * max(0, 3 - s['wins'] - s['draws'] - s['losses'])
        for t, s in standings_dict.items()
    }
    result = {}
    # Only meaningful when ≤2 matches remain (final matchday); more remaining
    # matches produce false positives because threat interactions become complex.
    final_matchday = len(remaining_matches) <= 2

    for rank, record in enumerate(ranked):
        team = record.team
        if not final_matchday:
            result[team] = {'confirmed_winner': False, 'confirmed_top2': False}
            continue

        cur_pts = record.points
        threats = {t for t, mp in max_pts.items() if t != team and mp >= cur_pts}

        # Count threat pairs who play each other but NOT against this team.
        # Only one of each such pair can simultaneously reach cur_pts.
        exclusive_pairs = 0
        seen: set = set()
        for t1, t2 in remaining_matches:
            pair = frozenset([t1, t2])
            if t1 in threats and t2 in threats and team not in (t1, t2) and pair not in seen:
                exclusive_pairs += 1
                seen.add(pair)

        min_simultaneous = len(threats) - exclusive_pairs
        result[team] = {
            'confirmed_winner': min_simultaneous == 0,
            'confirmed_top2': min_simultaneous <= 1,
        }
    return result


def _project_match(m1: Dict, m2: Dict, strengths: Dict, k: float) -> Dict:
    """Project a single knockout match from two prior results."""
    w1 = m1['team1'] if m1['team1_prob'] >= 0.5 else m1['team2']
    w2 = m2['team1'] if m2['team1_prob'] >= 0.5 else m2['team2']
    p = bradley_terry_win_prob(strengths.get(w1, 0.5), strengths.get(w2, 0.5), k)
    return {'team1': w1, 'team2': w2, 'team1_prob': round(p, 4), 'team2_prob': round(1 - p, 4)}


# Official FIFA 2026 bracket structure (Article 12.6-12.9), 0-indexed R32 positions
# R32 match order: 1=2Avs2B, 2=1Evs3rd, 3=1Fvs2C, 4=1Cvs2F, 5=1Ivs3rd,
#                  6=2Evs2I, 7=1Avs3rd, 8=1Lvs3rd, 9=1Dvs3rd, 10=1Gvs3rd,
#                  11=2Kvs2L, 12=1Hvs2J, 13=1Bvs3rd, 14=1Jvs2H, 15=1Kvs3rd, 16=2Dvs2G
R16_PAIRINGS = [(1, 4), (0, 2), (3, 5), (6, 7), (10, 11), (8, 9), (13, 15), (12, 14)]
# R16 order: M89(R32[1]vR32[4]), M90(R32[0]vR32[2]), M91(R32[3]vR32[5]), M92(R32[6]vR32[7]),
#            M93(R32[10]vR32[11]), M94(R32[8]vR32[9]), M95(R32[13]vR32[15]), M96(R32[12]vR32[14])
QF_PAIRINGS = [(0, 1), (4, 5), (2, 3), (6, 7)]
# QF order: M97(R16[0]vR16[1]), M98(R16[4]vR16[5]), M99(R16[2]vR16[3]), M100(R16[6]vR16[7])
SF_PAIRINGS = [(0, 1), (2, 3)]
# SF: M101(QF[0]vQF[1]), M102(QF[2]vQF[3])


@app.get("/api/bracket")
async def get_bracket(k: float = 0.5):
    """
    Return current group standings and expected R32 bracket with win probabilities.
    Teams/slots are flagged as confirmed once their group has played all 6 matches.
    Third-place slots require all 12 groups to be complete.
    """
    if simulator is None:
        raise HTTPException(status_code=500, detail="Simulator not initialized")

    groups = simulator.groups
    completed_matches = simulator.completed_matches
    all_groups_complete = all(completed_matches.has_group_completed(g) for g in groups)

    # Build per-group standings and rankings
    groups_data: Dict[str, Any] = {}
    for group_letter, teams in sorted(groups.items()):
        standings_dict = completed_matches.get_group_standings(group_letter, teams)
        match_results = completed_matches.get_group_match_results(group_letter)
        confirmed = completed_matches.has_group_completed(group_letter)

        records = {
            team: TeamRecord(
                team=team,
                points=standings_dict[team]['points'],
                goals_for=standings_dict[team]['goals_for'],
                goals_against=standings_dict[team]['goals_against'],
                wins=standings_dict[team]['wins'],
                draws=standings_dict[team]['draws'],
                losses=standings_dict[team]['losses'],
            )
            for team in teams
        }

        ranked = rank_group(records, match_results, simulator.fair_play, simulator.fifa_ratings)

        from simulation.match_loader import get_remaining_group_matches
        remaining = get_remaining_group_matches(group_letter, teams, completed_matches)
        conf_pos = _compute_confirmed_positions(standings_dict, ranked, remaining)
        groups_data[group_letter] = {
            'confirmed': confirmed,
            'teams': [
                {
                    'team': r.team,
                    'points': r.points,
                    'played': r.wins + r.draws + r.losses,
                    'wins': r.wins,
                    'draws': r.draws,
                    'losses': r.losses,
                    'goals_for': r.goals_for,
                    'goals_against': r.goals_against,
                    'goal_diff': r.goal_diff,
                    'confirmed_winner': conf_pos[r.team]['confirmed_winner'],
                    'confirmed_top2': conf_pos[r.team]['confirmed_top2'],
                }
                for r in ranked
            ],
        }

    # Build expected R32 bracket from current standings
    winners = {g: groups_data[g]['teams'][0]['team'] for g in groups_data}
    runners_up = {g: groups_data[g]['teams'][1]['team'] for g in groups_data}

    # Third-place: build TeamRecord objects for ranking across groups
    fake_standings: Dict[str, list] = {}
    for g, gdata in groups_data.items():
        fake_standings[g] = [
            TeamRecord(
                team=t['team'],
                points=t['points'],
                goals_for=t['goals_for'],
                goals_against=t['goals_against'],
                wins=t['wins'],
                draws=t['draws'],
                losses=t['losses'],
            )
            for t in gdata['teams']
        ]

    qualifying_groups, qualifying_teams, _ = get_third_place_qualifiers(
        fake_standings, simulator.fifa_ratings
    )

    # Build ranked third-place list for display
    third_place_records = {g: fake_standings[g][2] for g in fake_standings}
    third_place_ranked = rank_third_place_teams(third_place_records, simulator.fifa_ratings)
    third_place_data = [
        {
            'group': grp,
            'team': rec.team,
            'played': rec.wins + rec.draws + rec.losses,
            'points': rec.points,
            'wins': rec.wins,
            'draws': rec.draws,
            'losses': rec.losses,
            'goals_for': rec.goals_for,
            'goals_against': rec.goals_against,
            'goal_diff': rec.goal_diff,
            'qualifies': rec.team in qualifying_teams,
        }
        for grp, rec in third_place_ranked
    ]

    r32 = R32Bracket(winners, runners_up, qualifying_groups, qualifying_teams)

    # Determine which groups feed which R32 slot (for confirmed flag)
    # Winner slots: confirmed if that group is complete
    # Runner-up slots: same
    # Third-place slots: confirmed only when all groups complete (cross-group ranking)
    group_confirmed: Dict[str, bool] = {g: groups_data[g]['confirmed'] for g in groups_data}

    # Map each team in R32 to whether their slot is confirmed
    # Build a lookup: team -> confirmed for this specific slot
    # Winners and runners-up: confirmed if their group is done
    # Third-place qualifiers: confirmed only if all groups are done
    winner_confirmed = {g: group_confirmed[g] for g in groups_data}
    runner_up_confirmed = {g: group_confirmed[g] for g in groups_data}
    third_place_confirmed = all_groups_complete

    # confirmed_winner/top2 by group, for slot confirmation
    winner_conf = {g: groups_data[g]['teams'][0]['confirmed_winner'] for g in groups_data}
    runner_conf = {g: (groups_data[g]['teams'][1]['confirmed_top2'] and
                       groups_data[g]['teams'][0]['confirmed_winner']) for g in groups_data}

    def slot_confirmed(team: str) -> bool:
        for g, t in winners.items():
            if t == team:
                return group_confirmed[g] or winner_conf[g]
        for g, t in runners_up.items():
            if t == team:
                return group_confirmed[g] or runner_conf[g]
        return third_place_confirmed

    matchups_out = []
    for i, (t1, t2) in enumerate(r32.get_matchups()):
        s1 = simulator.strengths.get(t1, 0.5)
        s2 = simulator.strengths.get(t2, 0.5)
        prob1 = bradley_terry_win_prob(s1, s2, k)
        matchups_out.append({
            'match': i + 1,
            'team1': t1,
            'team2': t2,
            'team1_prob': round(prob1, 4),
            'team2_prob': round(1 - prob1, 4),
            'team1_confirmed': slot_confirmed(t1),
            'team2_confirmed': slot_confirmed(t2),
        })

    # Project R16 through Final using most-likely winners and official bracket structure
    str_map = simulator.strengths
    r16_matchups = [_project_match(matchups_out[a], matchups_out[b], str_map, k) for a, b in R16_PAIRINGS]
    qf_matchups = [_project_match(r16_matchups[a], r16_matchups[b], str_map, k) for a, b in QF_PAIRINGS]
    sf_matchups = [_project_match(qf_matchups[a], qf_matchups[b], str_map, k) for a, b in SF_PAIRINGS]
    final_matchup = _project_match(sf_matchups[0], sf_matchups[1], str_map, k)

    return {
        'groups': groups_data,
        'third_place_ranked': third_place_data,
        'r32_matchups': matchups_out,
        'r16_matchups': r16_matchups,
        'qf_matchups': qf_matchups,
        'sf_matchups': sf_matchups,
        'final_matchup': final_matchup,
        'all_groups_complete': all_groups_complete,
    }


@app.get("/api/third-place-probs")
async def get_third_place_probs(n: int = 2000):
    """
    Monte Carlo probability that each third-place team qualifies (top 8 of 12).
    Runs n group-stage simulations and counts qualification frequency.
    Only teams that are currently in 3rd place or could still finish 3rd are included.
    Returns all 12 current third-place teams sorted by qualification probability.
    """
    if simulator is None:
        raise HTTPException(status_code=500, detail="Simulator not initialized")

    n = min(max(n, 100), 10000)
    counts: Dict[str, int] = {}

    rng = np.random.default_rng()
    for _ in range(n):
        group_standings = simulate_all_groups(
            simulator.groups,
            simulator.strengths,
            simulator.fifa_ratings,
            simulator.mu,
            simulator.alpha,
            rng,
            simulator.completed_matches,
            fair_play=simulator.fair_play,
        )
        _, qualifying_teams, _ = get_third_place_qualifiers(group_standings, simulator.fifa_ratings)
        # Only count teams that finished 3rd in this simulation
        third_teams = {g: teams[2].team for g, teams in group_standings.items()}
        for team in qualifying_teams:
            if team in third_teams.values():
                counts[team] = counts.get(team, 0) + 1

    # Seed zero counts for teams not seen (e.g. locked-out groups)
    # Use one deterministic simulation to find the current 3rd-place team per group
    det_standings = simulate_all_groups(
        simulator.groups,
        simulator.strengths,
        simulator.fifa_ratings,
        simulator.mu,
        simulator.alpha,
        np.random.default_rng(0),
        simulator.completed_matches,
        fair_play=simulator.fair_play,
    )
    for g, teams in det_standings.items():
        if len(teams) >= 3 and teams[2].team not in counts:
            counts[teams[2].team] = 0

    result = sorted(
        [{'team': team, 'qualify_prob': round(c / n, 4)} for team, c in counts.items()],
        key=lambda x: -x['qualify_prob']
    )
    return {'probs': result, 'n_simulations': n}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "simulator_ready": simulator is not None,
        "num_teams": len(all_teams_list)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
