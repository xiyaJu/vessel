import numpy as np
from matplotlib.patches import Polygon, Circle

def animate_ship(ax, x, y, psi, loa, bol, cpa, color):
    L3 = 2 * loa / 3
    W2 = 2 * bol / 2

    x_mr = x - L3 * np.cos(psi)
    y_mr = y - L3 * np.sin(psi)

    x_ar = x_mr + W2 * np.sin(psi)
    y_ar = y_mr - W2 * np.cos(psi)

    x_al = x_mr - W2 * np.sin(psi)
    y_al = y_mr + W2 * np.cos(psi)

    x_fr = x + L3 * np.cos(psi) + W2 * np.sin(psi)
    y_fr = y + L3 * np.sin(psi) - W2 * np.cos(psi)

    x_fl = x + L3 * np.cos(psi) - W2 * np.sin(psi)
    y_fl = y + L3 * np.sin(psi) + W2 * np.cos(psi)

    x_fn = x + 2 * L3 * np.cos(psi)
    y_fn = y + 2 * L3 * np.sin(psi)

    vertices = np.array([
        [x_ar, y_ar],
        [x_fr, y_fr],
        [x_fn, y_fn],
        [x_fl, y_fl],
        [x_al, y_al]
    ])
    #print(f"[POLYGON] vertices=\n{vertices}")
    ship_polygon = Polygon(vertices, closed=True, edgecolor='black', facecolor=color)
    ax.add_patch(ship_polygon)
    #print(f"[POLYGON ADDED] total patches: {len(ax.patches)}")

def animate_static_obstacle(ax, xob, yob, cpa_ob, obs_col):
    inner_circle = Circle((xob, yob), cpa_ob, facecolor=obs_col, edgecolor='k')
    ax.add_patch(inner_circle)

    outer_circle = Circle((xob, yob), cpa_ob * 2, fill=False, linestyle=':', linewidth=1.5, edgecolor=obs_col)
    ax.add_patch(outer_circle)
