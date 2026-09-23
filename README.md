# NHL Expected Goals (xG) Model

A shot-quality ("expected goals") model for NHL shots, predicting an outcome (did this shot become a goal?)
from *where* on the ice a shot was taken combined with *when* in the game it
happened and the context at that moment (score, strength state).

Data comes directly from the NHL's public play-by-play API.

## Pipeline

1. **`src/fetch_data.py`** — pulls play-by-play JSON from the NHL API and
   extracts every shot attempt (shot-on-goal, goal, missed-shot, blocked-shot)
   into flat rows, tracking the running score as it goes.
2. **`src/features.py`** — engineers the spatio-temporal features:
   - *Spatial:* shot distance and angle to the net (from rink x/y coordinates)
   - *Temporal:* elapsed game time, period, score differential and strength
     state (5v5, power play, etc.) at the moment of the shot
3. **`src/model.py`** — trains a gradient-boosted classifier (XGBoost) and
   a logistic regression baseline on the identical train/test split, with
   standard evaluation metrics (ROC-AUC, log loss, Brier score) for both, for comparison.
4. **`src/visualize.py`** — plots shot locations colored by predicted xG, and
   a cumulative-xG-over-game-time chart showing when each team generated
   their scoring chances.

## Running it

```bash
pip install -r requirements.txt
jupyter notebook nhl_xg_model.ipynb
```

The notebook pulls a 2-week window by default (~100 games, ~12,000 shots),
which takes 2-3 minutes to fetch — increase `NUM_DAYS` for a bigger sample
(a full season is roughly 180 days), or decrease it for a faster run.

## Testing

```bash
pytest tests/
```

26 unit tests cover the pure logic that doesn't require live network calls:
the shot-distance/angle geometry, strength-state parsing, game-clock
conversion, and the play-by-play extraction/score-tracking logic (using a
small hand-built play-by-play fixture rather than hitting the live API).

## Results (sample run, ~12,000 shots from ~100 games, Jan 1-14, 2024)

| Model | ROC-AUC | Log loss | Brier score |
|---|---|---|---|
| XGBoost | 0.821 | 0.164 | 0.044 |
| Logistic Regression | 0.827 | 0.162 | 0.044 |

Goal rate in sample: ~5.0% (consistent with real NHL shooting percentages).

**The logistic regression baseline is not worse than XGBoost here** — if
anything it edges it out slightly. That's a legitimate result: with
only 5-6 well-chosen key features (distance, angle, game time, score state,
strength state), the relationship to goal probability is fairly smooth and
close to linear in log-odds space, which is exactly what logistic regression
is built to capture. Gradient boosting's advantage shows up more with larger
feature sets or messier, more interaction-heavy relationships. 




