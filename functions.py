import pandas as pd
import numpy as np
from sympy import symbols, Eq, solve
from matplotlib.font_manager import FontProperties
custom_font = FontProperties(fname="data/unicode.clarendb.ttf")

############################################################################################

# Distance between
# Compute the distance between two points


def distance_between(x1, x2, y1, y2):
    return abs(np.sqrt((x1-x2)**2+(y1-y2)**2))

############################################################################################

# Plot trajectory
# Compute the future trajectory of a player with a degree-3 polynomial

def plot_trajectory(data, length=1):
    data.index = range(len(data))
    cutting_frame = \
    data[data["event"].isin(["pass_outcome_caught", "handoff", "run", "lateral", "snap_direct", "fumble"])].index[0] + 1
    t = [0] + [i / 10 for i in range(1, cutting_frame)]
    x = list(data["x"])[0:cutting_frame]
    y = list(data["y"])[0:cutting_frame]
    a = list(data["a"])[0:cutting_frame]
    v = list(data["s"])[0:cutting_frame]

    # Degré du polynôme
    degree = 3

    # Interpolation polynomiale pour chaque variable
    coefficients_x = np.polyfit(t, x, degree)
    coefficients_y = np.polyfit(t, y, degree)
    coefficients_a = np.polyfit(t, a, degree)
    coefficients_v = np.polyfit(t, v, degree)

    # Prédiction pour de nouveaux points dans le temps
    t_future = [i / 10 for i in
                range(cutting_frame, cutting_frame + length)]  # Par exemple, prédire les 5 prochaines unités de temps
    x_future = np.polyval(coefficients_x, t_future)
    y_future = np.polyval(coefficients_y, t_future)
    a_future = np.polyval(coefficients_a, t_future)
    v_future = np.polyval(coefficients_v, t_future)

    # Affichage des résultats
    return {"x_real": x, "x_predicted": x_future, "y_real": y, "y_predicted": y_future,
            "x_full": data["x"][cutting_frame:cutting_frame + length],
            "y_full": data["y"][cutting_frame:cutting_frame + length]}

############################################################################################

# Intersection half lines
# Return if two half lines are crossing or not and their crossing point


def intersection_half_lines(start1, direction1, start2, direction2):
    # Paramètres pour les équations paramétriques des demi-droites
    t, u = symbols('t u')

    # Équations paramétriques des demi-droites
    line1 = [start1[i] + t * direction1[i] for i in range(len(start1))]
    line2 = [start2[i] + u * direction2[i] for i in range(len(start2))]

    # Conditions pour l'intersection
    intersection_condition = [Eq(line1[i], line2[i]) for i in range(len(line1))]

    # Résoudre les équations pour trouver les valeurs de t et u
    solution = solve(intersection_condition, (t, u))

    # Vérifier si les valeurs de t et u sont positives, ce qui indique une intersection
    if solution and all(value >= 0 for value in solution.values()):
        intersection_point = [start1[i] + solution[t] * direction1[i] for i in range(len(start1))]
        return True, intersection_point
    else:
        return False, None

############################################################################################

# Angle between vector
# Compute the angle between two vectors


def angle_entre_vecteurs(vecteur1, vecteur2):
    # Calculer le produit scalaire des deux vecteurs
    produit_scalaire = np.dot(vecteur1, vecteur2)

    # Calculer la norme de chaque vecteur
    norme_vecteur1 = np.linalg.norm(vecteur1)
    norme_vecteur2 = np.linalg.norm(vecteur2)

    # Calculer le produit scalaire normalisé
    produit_scalaire_normalise = produit_scalaire / (norme_vecteur1 * norme_vecteur2)

    # Calculer l'angle en radians
    angle_radians = np.arccos(produit_scalaire_normalise)

    # Convertir l'angle en degrés si nécessaire
    angle_degrees = np.degrees(angle_radians)

    return angle_degrees

############################################################################################

# Tackling position score
# Compute the TPS


def tackling_position_score(df_play, play_tracking_data, player_id):

    ########################## Get play's general informations ##########################

    # Play informations
    bcarrier = list(df_play["ballCarrierId"])[0]
    possTeam = list(df_play["possessionTeam"])[0]
    defTeam = list(df_play["defensiveTeam"])[0]

    # Get teams and player names
    ## Players
    players = list(pd.unique(play_tracking_data["nflId"]))
    ## Teams
    dico_teams = {"off": possTeam, "def": defTeam}

    # Df both teams and football at the moment of the catch by the ball carrier
    off_catch = play_tracking_data[(play_tracking_data["club"] == dico_teams["off"]) & (
        play_tracking_data["event"].isin(
            ["pass_outcome_caught", "handoff", "run", "lateral", "snap_direct", "fumble"]))]
    def_catch = play_tracking_data[(play_tracking_data["club"] == dico_teams["def"]) & (
        play_tracking_data["event"].isin(
            ["pass_outcome_caught", "handoff", "run", "lateral", "snap_direct", "fumble"]))]
    football_catch = play_tracking_data[(play_tracking_data["club"] == "football") & (play_tracking_data["event"].isin(
        ["pass_outcome_caught", "handoff", "run", "lateral", "snap_direct", "fumble"]))]

    ########################## Get ball carrier and tackler informations ##########################

    # Ball carrier informations
    bcar_x = list(off_catch[off_catch["nflId"] == bcarrier]["x"])[0]
    bcar_y = list(off_catch[off_catch["nflId"] == bcarrier]["y"])[0]
    data_traj_bcar = plot_trajectory(play_tracking_data[play_tracking_data["nflId"] == bcarrier])
    x_start_bcar = data_traj_bcar["x_real"][-1]
    x_end_bcar = data_traj_bcar["x_predicted"][0]
    y_start_bcar = data_traj_bcar["y_real"][-1]
    y_end_bcar = data_traj_bcar["y_predicted"][0]
    speed_bcar = list(off_catch[off_catch["nflId"] == bcarrier]["s"])[0]
    acceleration_bcar = list(off_catch[off_catch["nflId"] == bcarrier]["a"])[0]

    # Tackler informations
    tckl_x = list(def_catch[def_catch["nflId"] == player_id]["x"])[0]
    tckl_y = list(def_catch[def_catch["nflId"] == player_id]["y"])[0]
    data_traj_tckl = plot_trajectory(play_tracking_data[play_tracking_data["nflId"] == player_id])
    x_start_tckl = data_traj_tckl["x_real"][-1]
    x_end_tckl = data_traj_tckl["x_predicted"][0]
    y_start_tckl = data_traj_tckl["y_real"][-1]
    y_end_tckl = data_traj_tckl["y_predicted"][0]
    speed_tckl = list(def_catch[def_catch["nflId"] == player_id]["s"])[0]
    acceleration_tckl = list(def_catch[def_catch["nflId"] == player_id]["a"])[0]

    ######################### Compute ball-carrier - tackler score ################################

    distance = distance_between(bcar_x, tckl_x, bcar_y, tckl_y)

    # Elements for the metric
    ## Delta speed
    delta_speed = speed_bcar - speed_tckl
    ## Delta acceleration
    delta_acceleration = acceleration_bcar - acceleration_tckl
    ## Angle
    ### Check if the half line are crossing
    res_crossing = intersection_half_lines([x_start_tckl, y_start_tckl],
                                           [x_end_tckl - x_start_tckl, y_end_tckl - y_start_tckl],
                                           [x_start_bcar, y_start_bcar],
                                           [x_end_bcar - x_start_bcar, y_end_bcar - y_start_bcar])

    ### Constant value for crossing or not
    c = 0
    angle_bcar_tckl = 0
    opp_angle_bcar_tckl = 0
    dist_tckl_crossing = 1
    dist_bcar_crossing = 1
    if res_crossing[0] == True:
        c = 1
        pt_crossing = res_crossing[1]
        #### tackler - crossing point distance
        dist_tckl_crossing = distance_between(tckl_x, float(pt_crossing[0]), tckl_y, float(pt_crossing[1]))
        #### ball-carrier - crossing point distance
        dist_bcar_crossing = distance_between(bcar_x, float(pt_crossing[0]), bcar_y, float(pt_crossing[1]))

    ### Compute the angle
    if c == 1:
        angle_bcar_tckl = angle_entre_vecteurs([x_end_tckl - x_start_tckl, y_end_tckl - y_start_tckl],
                                               [x_end_bcar - x_start_bcar, y_end_bcar - y_start_bcar])
    if c == 0:
        opp_angle_bcar_tckl = angle_entre_vecteurs([x_start_tckl - x_end_tckl, y_start_tckl - y_end_tckl],
                                                   [x_start_bcar - x_end_bcar, y_start_bcar - y_end_bcar])

    # Score
    score_bcar_tckl = np.exp(distance / 3) + (
                max(0, delta_speed) + max(0, delta_acceleration) + (c * ((180 - angle_bcar_tckl) / 10)) + (
                    (1 - c) * (((180 - opp_angle_bcar_tckl) * distance) / 10)))

    ########################## Define players at risk, get informations about them, and compute score about them ##########################

    # Distance between the ball carrier and the tackler

    # Get players at risk
    off_players_at_risk = []
    for pId, x, y in zip(off_catch["nflId"], off_catch["x"], off_catch["y"]):
        if pId not in [bcarrier, player_id]:
            if (distance_between(x, bcar_x, y, bcar_y) < 5) or (distance_between(x, tckl_x, y, tckl_y) < 5):
                off_players_at_risk.append(pId)

    final_off_score = 0
    for player in off_players_at_risk:

        data = plot_trajectory(play_tracking_data[play_tracking_data["nflId"] == player])
        x_start = data["x_real"][-1]
        x_end = data["x_predicted"][0]
        y_start = data["y_real"][-1]
        y_end = data["y_predicted"][0]

        # Check if the players are crossing each others
        result = intersection_half_lines([x_start, y_start], [x_end - x_start, y_end - y_start],
                                         [x_start_tckl, y_start_tckl],
                                         [x_end_tckl - x_start_tckl, y_end_tckl - y_start_tckl])
        if result[0] == True:
            # Time for defender
            dist_x_tckl = float(result[1][0] - x_start_tckl)
            dist_y_tckl = float(result[1][1] - y_start_tckl)
            dist_tckl = np.sqrt((dist_x_tckl) ** 2 + (dist_y_tckl) ** 2)
            # time_tckl = calculer_temps(dist_tckl, speed_tckl, acceleration_tckl)

            # Time for offensive player
            speed_off = list(off_catch[off_catch["nflId"] == player]["s"])[0]
            acceleration_off = list(off_catch[off_catch["nflId"] == player]["a"])[0]
            dist_x_off = float(result[1][0] - x_start)
            dist_y_off = float(result[1][1] - y_start)
            dist_off = np.sqrt((dist_x_off) ** 2 + (dist_y_off) ** 2)
            # time_off = calculer_temps(dist_off, speed_off, acceleration_off)

            # Elements for the metric
            distance_tckl_off = round(
                np.sqrt((float(x_start - x_start_tckl)) ** 2 + (float(y_start - y_end_tckl)) ** 2), 3)
            delta_speed = round(speed_off - speed_tckl, 3)
            delta_acceleration = round(acceleration_off - acceleration_tckl, 3)
            angle = angle_entre_vecteurs([dist_x_tckl, dist_y_tckl], [dist_x_off, dist_y_off])
            score_off = ((max(0, delta_speed) + max(0, delta_acceleration) + (angle / 10)) / distance_tckl_off) * 10

            final_off_score += score_off
    final_score = score_bcar_tckl + final_off_score
    if final_score > 100:
        final_score = 100

    return final_score

############################################################################################

# Draw field
# Draw a football field


def draw_field(ax):
    ax.set_axis_off()
    ax.fill([0,0,120,120],[0,53.3,53.3,0], color="green")
    ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for x in range(0, 121, 10):
            ax.axvline(x, color='white')

    for x in range(10, 111, 1):
            ax.plot([x, x], [0,1], color="white")
            ax.plot([x, x], [53.3-1.1,53.3], color="white")
            ax.plot([x, x], [18.25-0.33, 18.25+0.33], color="white")
            ax.plot([x, x], [53.3-18.25+0.33, 53.3-18.25-0.33], color="white")

    for i, j in zip([1, 2, 3, 4, 5, 4, 3, 2, 1], [20, 30, 40, 50, 60, 70, 80, 90, 100]) :
        ax.text(j-2.5, 8, str(i), color="white", fontsize=25, fontproperties=custom_font)
        ax.text(j+0.5, 8, "0", color="white", fontsize=25, fontproperties=custom_font)
        ax.text(j+0.5, 53.3-10, str(i), color="white", fontsize=25, fontproperties=custom_font, rotation=180)
        ax.text(j-3, 53.3-10, "0", color="white", fontsize=25, fontproperties=custom_font, rotation=180)

    for i in [20,30,40,50] :
        ax.fill([i-3.5, i-4.5, i-3.5], [8.9, 9.2, 9.5], alpha=1, color='white')
        ax.fill([i-3.5, i-4.5, i-3.5], [53.3-9.5, 53.3-9.7, 53.3-10], alpha=1, color='white')

    for i in [70, 80, 90, 100] :
        ax.fill([i+3.5, i+4.5, i+3.5], [8.9, 9.2, 9.5], alpha=1, color='white')
        ax.fill([i+3.5, i+4.5, i+3.5], [53.3-9.5, 53.3-9.7, 53.3-10], alpha=1, color='white')
