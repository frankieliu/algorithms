import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Initial parameters
tokens = ["sunny", "cloudy", "rainy", "warm", "cold", "snowy", "windy", "humid"]
initial_logits = np.array([5.0, 4.2, 3.5, 3.0, 1.5, 2.0, 2.5, 1.8])
initial_temperature = 1.0
initial_num_tokens = 5

def softmax(logits, temperature):
    if temperature <= 0.01:
        probs = np.zeros_like(logits)
        probs[np.argmax(logits)] = 1.0
        return probs

    e_x = np.exp(logits / temperature)
    return e_x / e_x.sum()

# Create the figure and axes
fig = plt.figure(figsize=(14, 8))
ax_plot = plt.axes([0.1, 0.35, 0.8, 0.55])

# Create slider axes
ax_temp = plt.axes([0.15, 0.25, 0.7, 0.02])
ax_num_tokens = plt.axes([0.15, 0.20, 0.7, 0.02])

# Create sliders for individual logits
logit_sliders = []
logit_axes = []
for i in range(len(tokens)):
    ax = plt.axes([0.15, 0.14 - i*0.018, 0.7, 0.015])
    logit_axes.append(ax)

# Initialize sliders
temp_slider = Slider(ax_temp, 'Temperature', 0.01, 3.0, valinit=initial_temperature, valstep=0.05)
num_tokens_slider = Slider(ax_num_tokens, 'Num Tokens', 1, len(tokens), valinit=initial_num_tokens, valstep=1, valfmt='%d')

for i, (token, logit) in enumerate(zip(tokens, initial_logits)):
    slider = Slider(logit_axes[i], f'{token}', 0.0, 10.0, valinit=logit, valstep=0.1)
    logit_sliders.append(slider)

def update(val):
    # Get current values
    temperature = temp_slider.val
    num_tokens = int(num_tokens_slider.val)

    # Get current logits from sliders
    current_logits = np.array([slider.val for slider in logit_sliders[:num_tokens]])
    current_tokens = tokens[:num_tokens]

    # Calculate probabilities
    probs = softmax(current_logits, temperature)

    # Clear and redraw
    ax_plot.clear()
    bars = ax_plot.bar(current_tokens, probs, color='#4e79a7')
    ax_plot.set_ylim(0, 1.1)
    ax_plot.set_ylabel('Probability', fontsize=12)
    ax_plot.set_title(f'Probability Distribution at Temperature T = {temperature:.2f}', fontsize=14, fontweight='bold')

    # Annotate bars with percentage
    for bar in bars:
        height = bar.get_height()
        ax_plot.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.1%}', ha='center', va='bottom', fontsize=10)

    # Show/hide logit sliders based on num_tokens
    for i in range(len(tokens)):
        if i < num_tokens:
            logit_axes[i].set_visible(True)
        else:
            logit_axes[i].set_visible(False)

    fig.canvas.draw_idle()

# Connect sliders to update function
temp_slider.on_changed(update)
num_tokens_slider.on_changed(update)
for slider in logit_sliders:
    slider.on_changed(update)

# Initial plot
update(None)

plt.show()