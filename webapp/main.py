"""FastAPI web application for World Cup 2026 Bingo simulation."""
import math
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from simulation.simulator import WorldCupSimulator
from simulation.data_loader import load_all_data, get_all_teams


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
        le=2.0
    )
    k: float = Field(
        default=0.5,
        description="Bradley-Terry exponent parameter",
        ge=0.0,
        le=3.0
    )


class BingoCardResponse(BaseModel):
    """Response model for bingo simulation."""
    num_simulations: int
    parameters: Dict[str, float]
    results: List[Dict[str, Any]]


class TeamsListResponse(BaseModel):
    """Response model for available teams."""
    teams: List[str]


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

    # Validate cards
    for i, card in enumerate(request.cards):
        if len(card) != 18:
            raise HTTPException(
                status_code=400,
                detail=f"Card {i+1} must have exactly 18 teams, got {len(card)}"
            )

        # Check that all teams are valid
        invalid_teams = [t for t in card if t not in all_teams_list]
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
            request.cards,
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

    return BingoCardResponse(
        num_simulations=request.num_simulations,
        parameters={
            'mu': request.mu,
            'alpha': request.alpha,
            'k': request.k
        },
        results=results
    )


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
