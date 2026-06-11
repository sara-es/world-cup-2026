# World Cup 2026 Bingo Monte Carlo Simulator

A vibe coded Monte Carlo simulation system for analyzing World Cup 2026 knockout bingo cards. Simulates tournament outcomes to calculate the probability of all teams being knocked out (and bingo card completion) at each stage.

## Features

- **Monte Carlo Simulation**: Run thousands of tournament simulations with configurable parameters
- **ELO + FIFA Ratings**: Blended team strength model (60% ELO + 40% FIFA)
- **Poisson Goal Model**: Group stage matches simulated with Poisson distribution
- **Bradley-Terry Knockout**: Knockout matches use Bradley-Terry logistic model
- **Annex C Integration**: Accurate Round of 32 bracket allocation using FIFA's 495-scenario lookup table
- **Parameter Tuning**: Grid search over α (Poisson strength) and k (Bradley-Terry exponent)
- **Web Interface**: FastAPI backend with simple HTML frontend
- **CLI Support**: Command-line interface for batch simulations

## Project Structure

```
world-cup-2026/
├── simulation/              # Core simulation modules
│   ├── data_loader.py      # Load ELO, FIFA, group data
│   ├── team_strength.py    # Normalize & blend ratings
│   ├── group_stage.py      # Poisson match simulation
│   ├── third_place.py      # Third-place team ranking
│   ├── bracket.py          # R32 bracket construction
│   ├── knockout.py         # Bradley-Terry knockout
│   ├── bingo_tracker.py    # Track card completion
│   └── simulator.py        # Main orchestrator
├── webapp/                  # Web application
│   ├── main.py             # FastAPI backend
│   └── static/
│       └── index.html      # Frontend UI
├── main.py                  # CLI entry point
├── requirements.txt         # Dependencies
├── elo_ratings.tsv          # ELO ratings data
├── fifa_ratings.csv         # FIFA rankings data
├── bracket.csv              # Group assignments
└── annexe_c.py              # FIFA Annex C lookup table
```

## Installation

1. **Install dependencies**:
   ```bash
   uv add -r requirements.txt
   ```

2. **Verify data files are present**:
   - `elo_ratings.tsv`
   - `fifa_ratings.csv`
   - `bracket.csv`
   - `annexe_c.py`

## Usage

### Command Line Interface

**Run with a single card** (18 teams):
```bash
uv run -m main --teams Mexico Brazil Spain France Germany England Argentina Portugal \
                        Netherlands Belgium Japan USA Canada Australia Morocco \
                        Senegal Turkey Croatia
```

**Run with cards from JSON file**:
```bash
uv run -m main --cards my_cards.json -n 10000
```

**Adjust simulation parameters**:
```bash
uv run -m main --teams [18 teams...] -n 50000 --alpha 0.7 --k 2.0
```

**Run grid search**:
```bash
uv run -m main --cards my_cards.json --grid-search --verbose
```

**Save results to JSON**:
```bash
uv run -m main --teams [18 teams...] --output results.json
```

### Web Application

1. **Start the server**:
   ```bash
   uv run -m webapp.main
   ```

2. **Open browser**:
   - Navigate to `http://localhost:8000`
   - Select 18 teams from dropdowns
   - Configure parameters (optional)
   - Click "Run Simulation"

3. **API endpoints**:
   - `GET /api/teams` - List all 48 teams
   - `POST /api/simulate` - Run simulation (see example below)

**Example API request**:
```json
{
  "cards": [
    ["Mexico", "Brazil", "Spain", "France", "Germany", "England",
     "Argentina", "Portugal", "Netherlands", "Belgium", "Japan",
     "USA", "Canada", "Australia", "Morocco", "Senegal", "Turkey", "Croatia"]
  ],
  "num_simulations": 10000,
  "mu": 0.262,
  "alpha": 0.5,
  "k": 1.5
}
```

## Simulation Model

### Step 1: Team Strength Calculation
```
strength = 0.6 * normalize(ELO) + 0.4 * normalize(FIFA)
```

### Step 2: Group Stage (Poisson Model)
For each match between teams i and j:
```
λ_i = exp(μ + α * (strength_i - strength_j))
goals_i ~ Poisson(λ_i)

where:
  μ = ln(1.3) ≈ 0.262 (default), based on historical World Cup group stage average of 2.6 total goals per match, or 1.3 per team
  α = 0.35 (default, tunable), approx 73% chance that a team with 2 S.D.s higher strength rating will win
```

Rank teams by: **points → goal diff → goals scored → FIFA ranking**

### Step 3: Third-Place Selection
- Rank all 12 third-place teams by same criteria
- Select top 8
- Use **Annex C** to assign R32 bracket slots based on qualifying group combination

### Step 4: Knockout Stage (Bradley-Terry Model)
```
P(i beats j) = strength_i^k / (strength_i^k + strength_j^k)

where:
  k = 0.5 (default, tunable)
```

Simulate: **R32 → R16 → QF → SF → Final**

### Step 5: Bingo Tracking
For each card:
- Track teams eliminated at each stage
- Record first stage where all 18 teams are eliminated
- Aggregate across N simulations

## Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `mu` | ln(1.3) | - | Poisson base rate (avg goals per team) |
| `alpha` | 0.35 | 0.0 - 2.0 | Poisson strength effect (higher = more deterministic) |
| `k` | 0.5 | 0.0 - 3.0 | Bradley-Terry exponent (higher = more deterministic) |
| `num_sims` | 10,000 | 100 - 100,000 | Number of Monte Carlo simulations |
## Technical Details

### Annex C Lookup
The simulator uses FIFA's official 495-scenario lookup table from Annex C of the World Cup 2026 regulations. This ensures accurate Round of 32 bracket allocation based on which third-place teams qualify.

### Tiebreakers
Group stage ranking follows FIFA rules:
1. Points
2. Goal difference
3. Goals scored
4. ~~Disciplinary record~~ (not simulated due to laziness)
5. FIFA ranking

### Team Name Mapping
The data loader handles name variations:
- "USA" ↔ "United States"
- "IR Iran" ↔ "Iran"
- "Côte d'Ivoire" ↔ "Ivory Coast"
- etc.

## Known Limitations
1. **No disciplinary simulation**: Yellow/red cards not modeled (rarely affects third-place tiebreaker).

2. **Static team strengths**: Team strength doesn't change during tournament (no form/momentum modeling).

3. **Independent matches**: No correlation between matches (e.g., teams don't "tire" over tournament).

## License

This project is for educational and entertainment purposes.

## Acknowledgments

- ELO ratings data ([kaggle](https://www.kaggle.com/datasets/afonsofernandescruz/2026-fifa-world-cup-historical-elo-ratings))
- FIFA official rankings
- FIFA World Cup 2026 regulations (Annex C)

## Claude Code prompts

Someone at work has made a bingo raffle for the world cup. It has 18 teams on it, and the person whose teams are all eliminated first is the winner. If I wanted to calculate the odds of a certain card still being "active" at the end of each stage. I want to eventually make this a web app. First I want to write an MC sim that does:
for each simulation:
    simulate group stage -> determine who advances
    simulate R32 bracket -> determine who advances
    ... continue through to final
    for each card: record at which stage it completed
There are implementations that use the ELO data (60%) and the FIFA ratings (40%), which are not on the same scale so I'd have to normalize across all 48 teams and then compute blended strength.

Step 2: Match model
For a knockout match between i and j, win probability via logistic Bradley-Terry. Start with k=0.5 but I should be able to tune this. 

Step 3: Group stage simulation
For each of the 12 groups, simulate all matches by modelling each team's goals as independent poisson random variables. set mu=ln(1.3) and alpha=0.5 (related to strength difference) as a starting point.  Accumulate:
- Points (W=3, D=1, L=0)
- Goals scored and conceded

Step 4: Third-place qualification
After all 12 groups conclude, you have 12 third-place teams. Rank them by: points > goal difference > goals scored > disciplinary record > FIFA ranking. Ignore disciplinary record for simplicity in the simulation, but I should be able to include it for completed (deterministic) matches. Take the top 8. You need to keep track of which groups the 8 qualifying third-place teams came from, because this determines their R32 bracket slot.

Step 5: Knockout bracket (R32 through Final)
The R32 bracket has 16 fixed slots. Group winners and runners-up go into fixed positions; the 8 third-place teams slot in according to the group-combination table from Step 4. From R32 onwards it's single-elimination: simulate each match with the two-outcome model, advance winner, repeat.
The third-place match is not relevant for bingo purposes.

Step 6: Bingo card tracking
After each stage, check which of the 18 teams are still alive. Record the stage at which the card becomes complete (all 18 eliminated). Do this for all N simulations. Use these results to calculate the probability the card is active after stage s.

I should be able to run a grid search of alpha and k to be able to tune them. 

Notes: 
All files are in data/ 
1. ELO ratings are found in elo_ratings.tsv, FIFA rankings are fifa_ratings.csv, completed matches will go in matches.csv (this has only one test match for now, but I will add more as the tournament progresses), and the group assignments are in bracket.csv in order of group letter, first seed team, second seed team, third seed team, fourth seed team.
2. The bracket position table is annexe_c.py
3. Input should be a list of 18 teams by country name. I want a user to be able to input the country names on a web app and then get the output odds. There should also be a randomize button.
4. I should be able to tune the number of MC simulations in the web app, start with 10k as default
5. use python for the sim, fastapi for the webapp, the web app should be separate files to the sim logic
