import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Initial Data
probs = [0.9, 0.8, 0.95, 0.7] # Probabilities for 4 tokens
p_var = 0.5                  # Probability for the 5th token (variable)

def calculate_metrics(p_list):
    am = np.mean(p_list)
    gm = np.exp(np.mean(np.log(p_list)))
    perplexity = 1.0 / gm
    return am, gm, perplexity

fig, ax = plt.subplots(figsize=(8, 6))
plt.subplots_adjust(bottom=0.25)

# Initial Plot
names = ['Arith. Mean', 'Geom. Mean']
vals = calculate_metrics(probs + [p_var])[:2]
bars = ax.bar(names, vals, color=['skyblue', 'salmon'])
ax.set_ylim(0, 1)
ax.set_title(f"Perplexity: {calculate_metrics(probs + [p_var])[2]:.2f}")

# Add Slider
ax_p = plt.axes([0.2, 0.1, 0.65, 0.03])
s_p = Slider(ax_p, 'Prob of Token 5', 0.001, 1.0, valinit=p_var)

def update(val):
    current_p = s_p.val
    am, gm, perp = calculate_metrics(probs + [current_p])
    bars[0].set_height(am)
    bars[1].set_height(gm)
    ax.set_title(f"Impact of One Token: Perplexity = {perp:.2f}")
    fig.canvas.draw_idle()

s_p.on_changed(update)
plt.show()
