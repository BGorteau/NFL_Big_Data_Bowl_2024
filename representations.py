from functions import *

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
from matplotlib.legend_handler import HandlerPatch
import matplotlib.patches as mpatches
import math
from matplotlib import animation, rc
from matplotlib.patches import FancyArrowPatch
import numpy as np
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
from scipy import stats
from sympy import symbols, Eq, solve
import seaborn as sns

custom_font = FontProperties(fname="data/unicode.clarendb.ttf")

# Dataframes

tracking_data = pd.read_csv("data/final_notebook/play_example_tracking_data.csv")
tackles = pd.read_csv("data/tackles.csv")
plays = pd.read_csv("data/plays.csv")
tackling_scores = pd.read_csv("data/final_notebook/df_tackling_score.csv")
play_information = pd.read_csv("data/final_notebook/play_example_information.csv")
plys = pd.read_csv("data/players.csv")
sample_scores = pd.read_csv("data/final_notebook/sample_scores.csv")
sample_scores_u100 = sample_scores[sample_scores["score"]!=100]
sample_scores_100 = sample_scores[sample_scores["score"]==100]

###################################################################

# Informations about the play

bcarrier = list(play_information["ballCarrierId"])[0]
possTeam = list(play_information["possessionTeam"])[0]
defTeam = list(play_information["defensiveTeam"])[0]
players = list(pd.unique(tracking_data["nflId"]))
players = list(filter(lambda x: not math.isnan(x), players))

###################################################################

# Dataframes at catching moment

possTeam_catch = tracking_data[(tracking_data["club"]==possTeam)&(tracking_data["event"].isin(["pass_outcome_caught", "handoff", "run", "lateral", "snap_direct", "fumble"]))]
defTeam_catch = tracking_data[(tracking_data["club"]==defTeam)&(tracking_data["event"].isin(["pass_outcome_caught", "handoff", "run", "lateral", "snap_direct", "fumble"]))]
football_catch = tracking_data[(tracking_data["club"]=="football")&(tracking_data["event"].isin(["pass_outcome_caught", "handoff", "run", "lateral", "snap_direct", "fumble"]))]

# Ball carrier coordinates
bcar_x = list(possTeam_catch[possTeam_catch["nflId"]==bcarrier]["x"])[0]
bcar_y = list(possTeam_catch[possTeam_catch["nflId"]==bcarrier]["y"])[0]

# Tackler coordinates
tckl_x = list(defTeam_catch[defTeam_catch["nflId"]==52971]["x"])[0]
tckl_y = list(defTeam_catch[defTeam_catch["nflId"]==52971]["y"])[0]

# Offensive players at risk
off_players_at_risk = []
for pId, x, y in zip(possTeam_catch["nflId"], possTeam_catch["x"], possTeam_catch["y"]) :
    if pId not in [bcarrier, 52971] :
        if (distance_between(x, bcar_x, y, bcar_y) < 5) or (distance_between(x, tckl_x, y, tckl_y) < 5) :
            off_players_at_risk.append(pId)


###################################################################

# Represent players directions

fig, ax = plt.subplots(figsize=(10,10))
draw_field(ax)
plt.scatter(possTeam_catch["x"], possTeam_catch["y"], color="red", label="{} (POSS)".format(possTeam), zorder=4)
plt.scatter(defTeam_catch["x"], defTeam_catch["y"], color="blue", label="{} (DEF)".format(defTeam), zorder=4)
plt.scatter(football_catch["x"], football_catch["y"], color="black", label="Ball", zorder=4)

# Add an arrow for the direction of each player at risk
for player in players :
    data = plot_trajectory(tracking_data[tracking_data["nflId"]==player])
    x_start = data["x_real"][-1]
    x_end = data["x_predicted"][0]
    y_start = data["y_real"][-1]
    y_end = data["y_predicted"][0]
    ax.arrow(x_start, y_start, (x_end - x_start)*3, (y_end - y_start)*3, width=0.05,head_width=0.4, head_length=0.4, fc='black', ec='black', zorder=3)

#ax.arrow(0,0,0.2,0.2, width=0.01,head_width=0.03, head_length=0.03, fc='black', ec='black', label="$\overrightarrow{d_p}$ x 3")
plt.scatter(0,0,marker='>', color="black", label="$\overrightarrow{d_p}$ x 3")

plt.xlim(30,70)
plt.ylim(5,45)
plt.title("short-term predicted direction $\overrightarrow{d_p}$ of players at $Cm$", fontsize=20, weight="bold")
plt.legend(loc="upper right", fontsize=15)
plt.savefig("representations/players_directions.png", dpi=300)

###################################################################

# Represent at risk player
fig, ax = plt.subplots(figsize=(8,8))
draw_field(ax)
plt.scatter(possTeam_catch["x"], possTeam_catch["y"], color="red", label="{} (POSS)".format(possTeam),zorder=3)
plt.scatter(defTeam_catch["x"], defTeam_catch["y"], color="blue", label="{} (DEF)".format(defTeam),zorder=3)
plt.scatter(football_catch["x"], football_catch["y"], color="black",zorder=3)

# Plot black circles around the players at risk
df_players_at_risk = possTeam_catch[possTeam_catch["nflId"].isin(off_players_at_risk)]
plt.plot(tckl_x, tckl_y, marker='o', color="orange", markersize=8, markerfacecolor='none', markeredgewidth=2, linestyle='none', label="potential tackler", zorder=4)
plt.plot(bcar_x, bcar_y, marker='o', color="green", markersize=8, markerfacecolor='none', markeredgewidth=2, linestyle='none', label="ball carrier", zorder=4)
plt.plot(df_players_at_risk["x"], df_players_at_risk["y"], marker='o', color="black", markersize=8, markerfacecolor='none', markeredgewidth=2, linestyle='none', label="at-risk offense player", zorder=4)

distance=5

# Plot the circles around the ballcarrier and the tackler
theta = np.linspace(0, 2*np.pi, 100)
## Ball carrier
x = bcar_x + distance * np.cos(theta)
y = bcar_y + distance * np.sin(theta)
plt.fill(x, y, color="red", alpha=0.3, zorder=3, label="5 yards around tackler")

## Tackler
x_t = tckl_x + distance * np.cos(theta)
y_t = tckl_y + distance * np.sin(theta)
plt.fill(x_t, y_t, color="blue", alpha=0.3, zorder=3, label="5 yards around ball carrier")

plt.xlim(30,70)
plt.ylim(5,45)
plt.legend(loc="upper right", fontsize=12)
plt.title("At-risk offense players (first filtering) \n at $Cm$", fontsize=20, weight="bold")
plt.savefig("representations/at_risk_players.png", dpi=300)

###################################################################

# At risk player schema

fig, ax = plt.subplots(figsize=(12,8))
ax.set_axis_off()
plt.scatter([1], [1], color="blue", s=50, zorder=3)
plt.scatter([1], [2], color="red", s=50, zorder=3)
plt.scatter([6], [1], color="blue", s=50, zorder=3)
plt.scatter([6], [2], color="red", s=50, zorder=3)
ax.arrow(1, 1, -0.3, 0, width=0.01,head_width=0.05, head_length=0.05, fc='black', ec='black', zorder=2)
ax.arrow(1, 2, -0.3, -0.3, width=0.01,head_width=0.05, head_length=0.05, fc='black', ec='black', zorder=2)
ax.arrow(6, 1, 0.3, -0.3, width=0.01,head_width=0.05, head_length=0.05, fc='black', ec='black', zorder=2)
ax.arrow(6, 2, -0.3, -0.3, width=0.01,head_width=0.05, head_length=0.05, fc='black', ec='black', zorder=2)
plt.plot([0,1],[1,1], color="green", linestyle="--", zorder=1)
plt.plot([0,1],[1,2], color="green", linestyle="--", zorder=1)
plt.plot([5,6],[1,2], color="green", linestyle="--", zorder=1)
plt.plot([6,7],[1,0], color="green", linestyle="--", zorder=1)
plt.scatter(0,1,color="black", s=50, zorder=3)
plt.text(1.1,1, "Player A", fontsize=12)
plt.text(1.1,2, "Player B", fontsize=12)
plt.text(6.1,1, "Player A", fontsize=12)
plt.text(6.1,2, "Player B", fontsize=12)
plt.text(0.7,2.2, "Player A at risk", fontsize=15, horizontalalignment="center", fontweight="bold")
plt.text(5.5,2.2, "Player A not at risk", fontsize=15, horizontalalignment="center", fontweight="bold")
plt.xlim(-1,7)
plt.ylim(0,2.5)
plt.savefig("representations/draw_ARP.png", dpi=300)

###################################################################

# Represent score distribution

plt.subplots(figsize=(12,6))
ax = sample_scores["score"].hist(grid=False,
                        xlabelsize=10,
                        ylabelsize=12,
                        bins=50,
                        edgecolor='black',
                        color='orange',
                        rwidth=0.8
                        )
ax.set_title('Distribution of the scores (1000 miss and made tackles sample)',
                weight='bold',
                fontsize=20) 
ax.set_xlabel('$TPS$', fontsize=15)
ax.set_ylabel('Frequency', fontsize=15)
plt.savefig("representations/TPS_distribution.png", dpi=300)

###################################################################

# Yards after reception distribution

plt.subplots(figsize=(12,6))
tackles_u100 = sample_scores_u100[(sample_scores_u100["tackle"]==1)|(sample_scores_u100["assist"]==1)]
tackles_100 = sample_scores_100[(sample_scores_100["tackle"]==1)|(sample_scores_100["assist"]==1)]
sns.histplot(data=tackles_100, x="YardsAfterRec", color="red", kde=True, label="Tackles with $TPS$ = 100")
sns.histplot(data=tackles_u100, x="YardsAfterRec", color="blue", kde=True, label="Tackles with $TPS$ < 100")
plt.legend(loc="upper right", fontsize=12)
plt.xlabel("Yards after reception", fontsize=15)
plt.ylabel("Frequency", fontsize=15)
plt.title("Histograms of the yards after reception \n (sample of 883 made and assist tackles)", fontsize=20, fontweight="bold")
plt.savefig("representations/YAR_distribution.png", dpi=300)

###################################################################

# Represent ball catching

df_play = play_information
play_tracking_data = tracking_data

# Play informations
possTeam = list(df_play["possessionTeam"])[0]
defTeam = list(df_play["defensiveTeam"])[0]
game_id = list(df_play["gameId"])[0]
play_id = list(df_play["playId"])[0]

# Get tacklers id
tacklers_id = list(tackles[(tackles["gameId"]==game_id)&(tackles["playId"]==play_id)]["nflId"])

# Get teams and player names
## Players
players = list(pd.unique(play_tracking_data["nflId"]))
## Teams
dico_teams = {"off":possTeam, "def":defTeam}

# Df both teams and football at the moment of the catch by the ball carrier
off_catch = play_tracking_data[(play_tracking_data["club"]==dico_teams["off"])&(play_tracking_data["event"].isin(["pass_outcome_caught", "handoff", "run", "lateral", "snap_direct", "fumble"]))]
def_catch = play_tracking_data[(play_tracking_data["club"]==dico_teams["def"])&(play_tracking_data["event"].isin(["pass_outcome_caught", "handoff", "run", "lateral", "snap_direct", "fumble"]))]
football_catch = play_tracking_data[(play_tracking_data["club"]=="football")&(play_tracking_data["event"].isin(["pass_outcome_caught", "handoff", "run", "lateral", "snap_direct", "fumble"]))]
tacklers_catch = play_tracking_data[(play_tracking_data["nflId"]==tacklers_id[0])&(play_tracking_data["event"].isin(["pass_outcome_caught", "handoff", "run", "lateral", "snap_direct", "fumble"]))]

# Get scores for every player in defense
defense_scores = {}
for player in list(def_catch["nflId"]) :
    #print(player, ":", tackling_position_score(game_id, play_id, player))
    defense_scores[player] = tackling_position_score(df_play, play_tracking_data, player)
    
# Create a colormap
normalize = mcolors.Normalize(vmin=0, vmax=100)
colormap = plt.cm.hot
colors = {}
for key, val in defense_scores.items() :
    colors[key] = colormap(normalize(val))

fig, ax = plt.subplots(figsize=(9,8))
draw_field(ax)

# Add an arrow for offensive players
for player in off_catch["nflId"] :
    data = plot_trajectory(play_tracking_data[play_tracking_data["nflId"]==player])
    x_start = data["x_real"][-1]
    x_end = data["x_predicted"][0]
    y_start = data["y_real"][-1]
    y_end = data["y_predicted"][0]
    ax.arrow(x_start, y_start, (x_end - x_start)*3, (y_end - y_start)*3, width=0.05,head_width=0.4, head_length=0.4, fc='blue', ec='blue', zorder=3)

# Add an arrow for deffensive players
for player in def_catch["nflId"] :
    data = plot_trajectory(play_tracking_data[play_tracking_data["nflId"]==player])
    x_start = data["x_real"][-1]
    x_end = data["x_predicted"][0]
    y_start = data["y_real"][-1]
    y_end = data["y_predicted"][0]
    ax.arrow(x_start, y_start, (x_end - x_start)*3, (y_end - y_start)*3, width=0.05,head_width=0.4, head_length=0.4, fc='red', ec='red', zorder=3)

plt.scatter(off_catch["x"], off_catch["y"], color="blue", label="{} (POSS)".format(possTeam), zorder=3)
plt.scatter(def_catch["x"], def_catch["y"], color=list(colors.values()), zorder=3)
plt.scatter(0,0,color="red",label="{} (DEF)".format(defTeam))
plt.scatter(football_catch["x"], football_catch["y"], color="black", label="Ball",zorder=3)
plt.scatter(0,0,marker='>', color="red", label="$\overrightarrow{d_p}$ x 3 (DEF)")
plt.scatter(0,0,marker='>', color="blue", label="$\overrightarrow{d_p}$ x 3 (POSS)")
plt.plot(tacklers_catch["x"], tacklers_catch["y"], marker='o', color="black", markersize=8, markerfacecolor='none', markeredgewidth=2, linestyle='none', label="tackler(s)", zorder=4)

plt.xlim(list(football_catch["x"])[0]-20,list(football_catch["x"])[0]+20)
plt.ylim(list(football_catch["y"])[0]-20,list(football_catch["y"])[0]+20)
plt.legend(loc="upper right")
plt.title("TPS for defensive players \n (At catching moment $Cm$)", weight="bold", fontsize=20)

cbar = plt.colorbar(plt.cm.ScalarMappable(norm=normalize, cmap=colormap), ax=ax)
cbar.set_label('TPS for DEF players', fontsize=12)  # Vous pouvez personnaliser le label
plt.savefig("representations/ball_catching.png", dpi=300)

###################################################################

# Rankings

first_dict = {}
for pid in pd.unique(tackling_scores["playerId"]) :
    first_dict[pid] = []
    
for i in range(len(tackling_scores)) :
    if tackling_scores.loc[i,"tackle"] == 1 or tackling_scores.loc[i,"assist"] == 1 :
        first_dict[tackling_scores.loc[i,"playerId"]].append(tackling_scores.loc[i,"TPS"])

second_dict = {}
for key, val in first_dict.items():
    if len(val) > 30 :
        second_dict[key] = np.mean(val)
        
second_dict = dict(sorted(second_dict.items(), key=lambda item: item[1]))

names = [list(plys[plys["nflId"]==i]["displayName"])[0] for i in second_dict.keys()]
pos = [list(plys[plys["nflId"]==i]["position"])[0] for i in second_dict.keys()]
TPS = [round(i,2) for i in second_dict.values()]

df_ranking = pd.DataFrame({"Name" : names, "Position":pos, "TPS":TPS})


df_ranking_cb = df_ranking[df_ranking["Position"]=="CB"]
df_ranking_not_cb = df_ranking[df_ranking["Position"]!="CB"]

df_ranking_cb.index = range(len(df_ranking_cb))
df_ranking_not_cb.index = range(len(df_ranking_not_cb))

df_ranking_cb = df_ranking_cb.iloc[0:10]
df_ranking_not_cb = df_ranking_not_cb.iloc[0:10]

df_ranking_cb["Rank"] = range(1,11)
df_ranking_not_cb["Rank"] = range(1,11)

df_ranking_cb = df_ranking_cb.sort_values(by='TPS', ascending=False)
df_ranking_not_cb = df_ranking_not_cb.sort_values(by='TPS', ascending=False)


fig = plt.figure(figsize=(11,7), dpi=100)
ax = plt.subplot(121)

ncols = 4
nrows = 10

ax.set_xlim(0, ncols)
ax.set_ylim(0, nrows + 1)

positions = [0.25, 1.5, 2.5, 3.5]
columns = ["Rank", "Name", "Position", "TPS"]

# Add table's main text
for i in range(nrows):
    for j, column in enumerate(columns):
        if j == 0:
            ha = 'left'
        else:
            ha = 'center'
        if column == 'Rank':
            text_label = f'{df_ranking_cb[column].iloc[i]}'
            weight = 'bold'
        else:
            text_label = f'{df_ranking_cb[column].iloc[i]}'
            weight = 'normal'
        ax.annotate(
            xy=(positions[j], i + .5),
            text=text_label,
            ha=ha,
            va='center',
            weight=weight
        )

# Add column names
column_names = ["Rank", "Name", "Position", "TPS"]
for index, c in enumerate(column_names):
        if index == 0:
            ha = 'left'
        else:
            ha = 'center'
        ax.annotate(
            xy=(positions[index], nrows + .25),
            text=column_names[index],
            ha=ha,
            va='bottom',
            weight='bold'
        )

# Add dividing lines
ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [nrows, nrows], lw=1.5, color='black', marker='', zorder=4)
ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [0, 0], lw=1.5, color='black', marker='', zorder=4)
for x in range(1, nrows):
    ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [x, x], lw=1.15, color='gray', ls=':', zorder=3 , marker='')

ax.text(
    x=2, y=11,
    s='average TPS ranking for CB players \n (min 30 made ans assist tackles)',
    ha='center',
    va='bottom',
    weight='bold',
    size=12
)
ax.set_axis_off()

ax = plt.subplot(122)

ncols = 4
nrows = 10

ax.set_xlim(0, ncols)
ax.set_ylim(0, nrows + 1)

positions = [0.25, 1.5, 2.5, 3.5]
columns = ["Rank", "Name", "Position", "TPS"]

# Add table's main text
for i in range(nrows):
    for j, column in enumerate(columns):
        if j == 0:
            ha = 'left'
        else:
            ha = 'center'
        if column == 'Rank':
            text_label = f'{df_ranking_not_cb[column].iloc[i]}'
            weight = 'bold'
        else:
            text_label = f'{df_ranking_not_cb[column].iloc[i]}'
            weight = 'normal'
        ax.annotate(
            xy=(positions[j], i + .5),
            text=text_label,
            ha=ha,
            va='center',
            weight=weight
        )

# Add column names
column_names = ["Rank", "Name", "Position", "TPS"]
for index, c in enumerate(column_names):
        if index == 0:
            ha = 'left'
        else:
            ha = 'center'
        ax.annotate(
            xy=(positions[index], nrows + .25),
            text=column_names[index],
            ha=ha,
            va='bottom',
            weight='bold'
        )

# Add dividing lines
ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [nrows, nrows], lw=1.5, color='black', marker='', zorder=4)
ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [0, 0], lw=1.5, color='black', marker='', zorder=4)
for x in range(1, nrows):
    ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [x, x], lw=1.15, color='gray', ls=':', zorder=3 , marker='')

ax.text(
    x=2, y=11,
    s='average TPS ranking for non CB players \n (min 30 made and assist tackles)',
    ha='center',
    va='bottom',
    weight='bold',
    size=12
)
ax.set_axis_off()
plt.savefig("representations/rankings.png", dpi=300)
