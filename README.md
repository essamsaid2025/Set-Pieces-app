# Set Piece Analysis App

A Streamlit app for analyzing attacking and defensive set pieces with the same visual system and theme language as the original Football Charts Generator.

## Project structure

```text
set_piece_app/
├─ app.py
├─ data_utils.py
├─ ui_theme.py
├─ set_piece_charts.py
├─ requirements.txt
└─ assets/
```

## Features

* Same theme names as the original app:

  * The Athletic Dark
  * Opta Dark
  * Sofa Light
  * Black Stripe
* Left controls / right preview layout
* Chart requirement display under each chart
* PNG download for every chart
* PDF download for the full report
* Filters for set piece type, team, side, and delivery type

## Included charts

* Delivery Heatmap
* Delivery End Scatter
* Outcome Distribution
* Target Zone Breakdown
* First Contact Win By Zone
* Routine Breakdown
* Shot Map
* xG By Routine
* Second Ball Map
* Defensive Vulnerability Map
* Taker Profile
* Structure Zone Averages

## Required setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Run locally:

```bash
streamlit run app.py
```

## Recommended input columns

```text
match_id
team
opponent
date
competition
set_piece_type
side
delivery_type
taker
sequence_id
phase
x
y
x2
y2
target_zone
outcome
result
first_contact_win
second_ball_win
players_near_post
players_far_post
players_6yard
players_penalty
defenders_near_post
defenders_far_post
target_player
first_contact_player
shot_player
xg
routine_type
```

## Deployment

### GitHub

Create a new repository in the same GitHub account, then upload:

* app.py
* data_utils.py
* ui_theme.py
* set_piece_charts.py
* requirements.txt
* README.md

### Streamlit Community Cloud

1. Connect your GitHub account
2. Select the repository
3. Choose `app.py` as the main file
4. Deploy
5. Share the generated app link with your team

## Notes

* Use CSV or Excel input files
* Charts will render only when their required columns exist
* Missing columns are shown directly under each chart card
