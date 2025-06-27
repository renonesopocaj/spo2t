import sys
import numpy as np
import serial
import serial.tools.list_ports
from collections import deque
import re # Importa il modulo re
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QLabel, QComboBox,
                           QLCDNumber, QProgressBar)
from PyQt5.QtCore import QTimer, Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class ApneaMonitor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Monitoraggio Apnee - MAX30102")
        self.setGeometry(100, 100, 1200, 800)
        
        # Impostazioni per il plotting
        self.window_size = 100  # Mostra ultimi 100 campioni
        self.data_buffers = {
            'spo2': deque(maxlen=self.window_size),
            'prob': deque(maxlen=self.window_size),
            'time': deque(maxlen=self.window_size)
        }
        
        # Variabili per il conteggio apnee
        self.apnea_count = 0
        self.in_apnea_event = False
        self.time_last_apnea = 0  # ultimo timestamp (s) in cui è stata contata un'apnea
        
        # Setup seriale
        self.serial_port = None
        self.start_time = None
        self.setup_ui()
        
    def setup_ui(self):
        # Widget principale
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # Controlli superiori
        controls = QHBoxLayout()
        
        # Selezione porta
        port_group = QHBoxLayout()
        self.port_combo = QComboBox()
        self.refresh_ports_button = QPushButton("Aggiorna Porte")
        self.refresh_ports_button.clicked.connect(self.refresh_serial_ports)
        port_group.addWidget(QLabel("Porta:"))
        port_group.addWidget(self.port_combo)
        port_group.addWidget(self.refresh_ports_button)
        
        # Pulsante connessione
        self.connect_button = QPushButton("Connetti")
        self.connect_button.clicked.connect(self.toggle_connection)
        port_group.addWidget(self.connect_button)
        
        controls.addLayout(port_group)
        controls.addStretch()
        
        # Display numerici
        displays = QHBoxLayout()
        
        # SpO2 Display
        spo2_group = QVBoxLayout()
        spo2_label = QLabel("SpO2 (%)")
        spo2_label.setAlignment(Qt.AlignCenter)
        self.spo2_display = QLCDNumber()
        self.spo2_display.setDigitCount(5)
        self.spo2_display.setSegmentStyle(QLCDNumber.Flat)
        self.spo2_display.setMinimumHeight(80) # Altezza maggiore
        spo2_group.addWidget(spo2_label)
        spo2_group.addWidget(self.spo2_display)
        
        # Conteggio Apnee Display
        apnea_count_group = QVBoxLayout()
        apnea_count_label = QLabel("Numero apnee")
        apnea_count_label.setAlignment(Qt.AlignCenter)
        self.apnea_count_display = QLCDNumber()
        self.apnea_count_display.setDigitCount(5) # 5 cifre per il conteggio
        self.apnea_count_display.setSegmentStyle(QLCDNumber.Flat)
        self.apnea_count_display.setMinimumHeight(80) # Altezza maggiore
        apnea_count_group.addWidget(apnea_count_label)
        apnea_count_group.addWidget(self.apnea_count_display)
        
        displays.addLayout(spo2_group)
        displays.addLayout(apnea_count_group)
        
        layout.addLayout(controls)
        layout.addLayout(displays)
        
        # Setup dei grafici
        plots = QHBoxLayout()
        
        # Grafico SpO2
        self.figure_spo2 = Figure(figsize=(6, 4))
        self.canvas_spo2 = FigureCanvas(self.figure_spo2)
        self.ax_spo2 = self.figure_spo2.add_subplot(111)
        self.line_spo2, = self.ax_spo2.plot([], [], 'b-', label='SpO2')
        self.ax_spo2.set_title('SpO2 nel Tempo')
        self.ax_spo2.set_xlabel('Tempo (s)')
        self.ax_spo2.set_ylabel('SpO2 (%)')
        self.ax_spo2.grid(True)
        self.ax_spo2.legend()
        
        # Grafico Probabilità
        # self.figure_prob = Figure(figsize=(6, 4))
        # self.canvas_prob = FigureCanvas(self.figure_prob)
        # self.ax_prob = self.figure_prob.add_subplot(111)
        # self.line_prob, = self.ax_prob.plot([], [], 'r-', label='Probabilità')
        # self.ax_prob.set_title('Probabilità Apnea nel Tempo')
        # self.ax_prob.set_xlabel('Tempo (s)')
        # self.ax_prob.set_ylabel('Probabilità (%)')
        # self.ax_prob.set_ylim(0, 100)
        # self.ax_prob.grid(True)
        # self.ax_prob.legend()
        
        plots.addWidget(self.canvas_spo2)
        # plots.addWidget(self.canvas_prob)
        
        layout.addLayout(plots)
        
        # Timer per l'aggiornamento
        self.timer = QTimer()
        self.timer.setInterval(50)  # 50ms = 20Hz
        self.timer.timeout.connect(self.update_plot)
        
        # Inizializza la lista delle porte
        self.refresh_serial_ports()
        self.apnea_count_display.display(self.apnea_count) # Mostra conteggio iniziale
        
    def refresh_serial_ports(self):
        self.port_combo.clear()
        ports = [port.device for port in serial.tools.list_ports.comports()]
        if ports:
            self.port_combo.addItems(ports)
        
    def toggle_connection(self):
        if self.serial_port is None:
            try:
                port = self.port_combo.currentText()
                self.serial_port = serial.Serial(port, 115200, timeout=1)
                self.connect_button.setText("Disconnetti")
                self.start_time = None
                # Resetta il conteggio apnee e lo stato alla connessione
                self.apnea_count = 0
                self.in_apnea_event = False
                self.time_last_apnea = 0
                self.apnea_count_display.display(self.apnea_count)
                for key in self.data_buffers:
                    self.data_buffers[key].clear()
                self.timer.start()
            except Exception as e:
                print(f"Errore di connessione: {str(e)}")
        else:
            self.timer.stop()
            if self.serial_port and self.serial_port.is_open:
                self.serial_port.close()
            self.serial_port = None
            self.connect_button.setText("Connetti")
            
    def update_plot(self):
        if not self.serial_port or not self.serial_port.is_open:
            return
            
        try:
            # Leggi TUTTE le righe attualmente disponibili, così il buffer non si accumula
            while self.serial_port.in_waiting:
                line = self.serial_port.readline().decode('utf-8', errors='ignore').strip()

                # Evita di processare stringhe vuote
                if not line:
                    continue

                # Si attende una riga con tre valori: SpO2, probabilità, timestamp
                # I valori possono essere separati da spazi o virgole, es.: "97.8 0.34 123456" oppure "97.8,0.34,123456"
                # Sostituiamo le virgole con spazi e poi facciamo lo split
                tokens = line.replace(',', ' ').split()

                if len(tokens) >= 3:
                    spo2_str, prob_str, ts_str = tokens[:3]

                    # Inizializza il tempo di riferimento al primo sample
                    if self.start_time is None:
                        # Arduino millis() -> secondi
                        self.start_time = float(ts_str) / 1000.0

                    # Calcolo asse x come tempo relativo in secondi
                    current_time = (float(ts_str) / 1000.0) - self.start_time

                    try:
                        spo2 = float(spo2_str)
                        prob = float(prob_str) * 100.0  # convertiamo in percentuale
                    except ValueError:
                        return  # salta se parsing fallisce
                    
                    # Aggiorna i buffer
                    self.data_buffers['time'].append(current_time)
                    if spo2 == int(100):
                        # print("SpO2 = 100%")
                        spo2 = 99.00
                    self.data_buffers['spo2'].append(spo2)
                    self.data_buffers['prob'].append(prob)
                    
                    # Aggiorna i display numerici
                    self.spo2_display.display(f"{spo2:.1f}")

                    # Logica conteggio apnee (attiva solo dopo 40 s per stabilizzazione sensore)
                    if current_time >= 40:
                        # Verifica se è passato abbastanza tempo dall'ultima apnea
                        if current_time - self.time_last_apnea >= 15:
                            # Verifica il fronte di salita (crossing) della soglia
                            if prob > 50 and not self.in_apnea_event:
                                self.apnea_count += 1
                                self.in_apnea_event = True
                                self.time_last_apnea = current_time
                                self.apnea_count_display.display(self.apnea_count)
                            elif prob <= 50:
                                self.in_apnea_event = False
                    else:
                        # Prima dei 40 s azzera lo stato evento per evitare conteggi spuri
                        self.in_apnea_event = False
                        self.time_last_apnea = 0

                    # Aggiorna i grafici
                    time_data = list(self.data_buffers['time'])

                    # Aggiorna grafico SpO2
                    self.line_spo2.set_data(time_data, list(self.data_buffers['spo2']))
                    if time_data:
                        self.ax_spo2.set_xlim(time_data[0], max(time_data[-1], self.window_size / 20))
                    if self.data_buffers['spo2']:
                        self.ax_spo2.set_ylim(min(self.data_buffers['spo2']) - 1, max(self.data_buffers['spo2']) + 1)
                    self.canvas_spo2.draw()

                    # Aggiorna grafico Probabilità
                    # self.line_prob.set_data(time_data, list(self.data_buffers['prob']))
                    # if time_data:
                    #     self.ax_prob.set_xlim(time_data[0], max(time_data[-1], self.window_size / 20))
                    # self.canvas_prob.draw()
                    
        except Exception as e:
            print(f"Errore durante l'aggiornamento del plot: {str(e)}")
            print(f"Riga ricevuta: {line}") # Stampa la riga per debug
            
    def closeEvent(self, event):
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = ApneaMonitor()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 