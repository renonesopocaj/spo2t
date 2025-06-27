#!/usr/bin/env python3
"""Script stand-alone che visualizza un grafico Matplotlib dei valori SpO2 forniti,
spaziati di 1 s, e mostra a schermo il tempo di inferenza e la probabilità di apnea.
"""

import matplotlib.pyplot as plt
import numpy as np

# Dati forniti
spo2_values = [
    99, 99, 99, 99, 99, 99, 99, 98, 98, 98,
    97, 97, 96, 96, 97, 97, 98, 98, 99, 99,
    98, 99, 99, 99, 99,
]

# Tempo (1 s fra un campione e l'altro)
time_seconds = np.arange(len(spo2_values))

# Metadati fissi richiesti
inference_time_ms = 1  # ms
apnea_probability = 0.65  # esempio 0–1

# Creazione grafico
plt.style.use("seaborn-v0_8-darkgrid")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(time_seconds, spo2_values, marker="o", label="SpO2 (%)")
ax.set_xlabel("Tempo (s)")
ax.set_ylabel("SpO2 (%)")
ax.set_title("Trend SpO2 con finestra di 25 s")
ax.set_ylim(90, 100)
ax.legend()

# Testo informativo in basso
text = (
    f"Inference time: {inference_time_ms} ms\n"
    f"Apnea probability: {apnea_probability:.2%}"
)
# Coordinate (0.02, 0.05) -> in basso a sinistra
ax.text(
    0.02,
    0.05,
    text,
    transform=ax.transAxes,
    fontsize=11,
    verticalalignment="bottom",
    bbox=dict(boxstyle="round", fc="w", alpha=0.8),
)

plt.tight_layout()
plt.show()
