# SpO2t

OSA detection on the Nicla Sense ME, thanks to a MLP trained on 25-sample-windows of $\text{SpO}_2$ data at $1 \text{Hz}$, labeled by doctors from the Sismanoglio General Hospital of Athens.
## Software
- a `max30102-apnea-detection.ino` main file to collect $R$ and $IR$ samples from MAX30102, calculate $\text{SpO}_2$ and perform the inference. We implemented a rolling average as a low pass filter and a linear interpolation to exclude invalid measurements/outliers and fill the missing values. For demo purposes, due to the inaccurate measurements from MAX30102, we remove outliers quite "brutally".
- `gating_PPG_DEMO_2.py` to detect movements that might compromise MAX30102 measurements quality.
- A `dataset\` folder with the dataset that we obtained by pre-processing PSG-AUDIO (removing unwanted channels from the PSG, creating appropriate windows and assigning them the appropriate labels).
- `eda\` folder to to explorative data analysis on teh folder. 
- `cortex-m7\` folder containing the static library used to perform the inference on the Nicla.
- `scripts-for-demo\` folder containing the `hackathon-demo-noprob.py` script displaying OSA count and the time series of $\text{SpO}_2$ data collected from the MAX30102.
- you can generate `optuna.db` for the optuna dashboard.
- `requirements.txt` dependencies file for python scripts.
- All other python scripts are either to train/evaluate the mlp (`mlp_apnea_optuna.py`), or helpers for the training process (`prepare_csv_windows.py`), or to export the obtained trained single model (no ensembles) `export_onnx.py` (or the eda process for `add_apnea_events.py`, `eda_spo2_windows.py`).
- Some considerations on the MLP: we didn't use ensembles since they're too heavy for the Nicla. The MLP is extremely small. Achieved accuracy is of $\approx 80\%$. Dropout is used (even though the model is very small). Optuna was used for hyperparameters tuning, nn.BCEWithLogitsLoss() was used as a loss function, with AdamW as gradient optimizer. Early stopping and a learning rate scheduler was used.
- Zant SDK was used to run/deploy the MLP on the Nicla Sense ME, by generating an executable from the ONNX of the MLP. 

## Hardware
MAX30102, Nicla Sense ME.

## Contributors
- Jacopo Senoner jacopo.senoner@mail.polimi.it
- Michelangelo Stefanini michelangelo.stefanini@mail.polimi.it
- Alessandro Trimarchi alessandro.trimarchi@mail.polimi.it
