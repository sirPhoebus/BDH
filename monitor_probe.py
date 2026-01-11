import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import os
from datetime import datetime

# Configuration
PROBE_FILE = 'probe.json'
MAX_POINTS = 500  # How many data points to show on the graph
INTERVAL_MS = 100 # Poll every 100ms

# Data storage
data_history = {}  # Key -> List of values
steps = []

def read_probe():
    try:
        with open(PROBE_FILE, 'r') as f:
            content = f.read().strip()
            if not content:
                return None
            return json.loads(content)
    except (json.JSONDecodeError, FileNotFoundError, PermissionError):
        return None

def init():
    # Attempt to read the file once to get keys
    data = read_probe()
    if data:
        for key in data:
            if key != 'step' and isinstance(data[key], (int, float)):
                data_history[key] = []
        return True
    return False

# Initialize plots
fig = plt.figure(figsize=(12, 10))
fig.suptitle('Brain Simulation Telemetry', fontsize=16)

# We'll determine the layout dynamically based on number of fields
data = None
while data is None:
    data = read_probe()
    if data is None:
        print("Waiting for probe.json to be created/populated...")
        time.sleep(1)

# Filter for numeric keys
numeric_keys = [k for k, v in data.items() if k != 'step' and isinstance(v, (int, float))]
num_plots = len(numeric_keys)
cols = 2
rows = (num_plots + 1) // 2

axes = {}
lines = {}

for i, key in enumerate(numeric_keys):
    ax = fig.add_subplot(rows, cols, i + 1)
    ax.set_title(key)
    ax.grid(True, alpha=0.3)
    line, = ax.plot([], [], lw=2)
    lines[key] = line
    axes[key] = ax
    data_history[key] = []

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

def update(frame):
    current_data = read_probe()
    if not current_data:
        return lines.values()
    
    current_step = current_data.get('step', 0)
    
    # Avoid duplicate steps if file hasn't changed
    if steps and steps[-1] == current_step:
        return lines.values()
        
    steps.append(current_step)
    if len(steps) > MAX_POINTS:
        steps.pop(0)
        
    for key in numeric_keys:
        val = current_data.get(key, 0)
        data_history[key].append(val)
        if len(data_history[key]) > MAX_POINTS:
            data_history[key].pop(0)
            
        # Update line data
        lines[key].set_data(steps, data_history[key])
        
        # Rescale axes
        ax = axes[key]
        ax.relim()
        ax.autoscale_view()
        
    return lines.values()

print(f"Monitoring {PROBE_FILE}...")
ani = animation.FuncAnimation(fig, update, interval=INTERVAL_MS, blit=False)
plt.show()
