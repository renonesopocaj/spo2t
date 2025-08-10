/*
Questo script si occupa di acquisire i campioni dal max30102 e performare l'inferenza. Contiene vari print di debug e i print finali non sono stati tolti per compatibilità
con lo script hackathon-demo-noprob.py (o simili) che si occupano di mostrare a schermo il grafico della spo2 e la probabilità inferita dall'mlp di avere un apnea in tempo
reale.
*/

#include <Wire.h>
#include "MAX30105.h"
#include "spo2_algorithm.h"
#include <lib_zant.h>
#include <math.h>
#include <Arduino.h>
#include <Nicla_System.h>

// costanti del sensore
constexpr uint16_t SAMPLE_RATE   = 100;   
constexpr uint16_t RAW_BUF_LEN   = 50; 
constexpr uint8_t  N_SAMPLES     = 25;  

// normalizzazione analoga al modello
#define GLOBAL_MEAN_SPO2 93.815f
#define GLOBAL_STD_SPO2 8.207f

// sigmoide
static inline float sigmoidf(float x) {
  return 1.0f / (1.0f + expf(-x));
}

// finestre e statistiche
static float spo2_window[N_SAMPLES] = {0};
static uint8_t window_count = 0;

static uint64_t prep_acc_us  = 0;
static uint64_t infer_acc_us = 0;
static uint32_t loop_counter = 0;

// setup sensore e buffer r/ir
MAX30105 particleSensor;

#if defined(__AVR_ATmega328P__) || defined(__AVR_ATmega168__)
uint16_t irBuffer[RAW_BUF_LEN];
uint16_t redBuffer[RAW_BUF_LEN];
#else
uint32_t irBuffer[RAW_BUF_LEN];
uint32_t redBuffer[RAW_BUF_LEN];
#endif

int32_t spo2 = 0;
int8_t  validSPO2 = 0;
int32_t heartRate = 0;
int8_t  validHeartRate = 0;

int32_t countInvSpo2Med = 0;

// slider della finestra di spo2
void updateSpo2Window(float v) {
    // se la finestra NON è ancora piena sposta solo il cursore
    if (window_count < N_SAMPLES) {
      window_count++;
    }
    // shift 
    for (int8_t i = window_count - 1; i > 0; --i) {
      spo2_window[i] = spo2_window[i - 1];
    }
    spo2_window[0] = v;  
}


/* ───────── setup ───────── */
void setup() {
  nicla::begin();
  Serial.begin(115200);
  if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) {
    // Serial.println(F("MAX30105 not found"));
    while (1);
  }
  // setup max30102
  particleSensor.setup(60, 4, 2, SAMPLE_RATE, 411, 16384);
  // Serial.println(F("Warm up samples"));
}

void loop() {
  // ------------------------ WARM UP DEI BUFFER R, IR -----------------------------
  // Warm-up: riempi il buffer  
  for (uint16_t i = 0; i < RAW_BUF_LEN; ++i) {
    while (!particleSensor.available()) particleSensor.check();
    redBuffer[i] = particleSensor.getRed();
    irBuffer[i]  = particleSensor.getIR();
    particleSensor.nextSample();
  }

  // calcola il primo SpO2 
  maxim_heart_rate_and_oxygen_saturation(irBuffer, RAW_BUF_LEN,
                                         redBuffer,
                                         &spo2, &validSPO2,
                                         &heartRate, &validHeartRate);
  if (validSPO2) updateSpo2Window((float)spo2);

  // ------------------------ WARM UP DELLA WINDOW -----------------------------
  // continua finché la finestra SpO2 non è piena (25 s)
  while (window_count < N_SAMPLES) {

    while (!particleSensor.available()) particleSensor.check(); // attendi

    // scivola di 1 campione nel buffer raw 
    memmove(redBuffer, redBuffer + 1, (RAW_BUF_LEN - 1) * sizeof(redBuffer[0]));
    memmove(irBuffer,  irBuffer  + 1, (RAW_BUF_LEN - 1) * sizeof(irBuffer[0]));

    redBuffer[RAW_BUF_LEN - 1] = particleSensor.getRed();
    irBuffer [RAW_BUF_LEN - 1] = particleSensor.getIR();
    particleSensor.nextSample();

    maxim_heart_rate_and_oxygen_saturation(irBuffer, RAW_BUF_LEN, redBuffer, &spo2, &validSPO2, &heartRate, &validHeartRate);

    if (validSPO2 && spo2 > 95) updateSpo2Window((float)spo2); // balza sample se invalido

    // Serial.print(F("Warm-up "));
    // Serial.print(window_count);
    // Serial.print(F("/25  SpO2="));
    // Serial.println(spo2);
  }

  // Serial.println(F("Comincio inferenza"));

  // ------------------------------------- WHILE INFINITO -------------------------------------
  // flusso: acquisisco campione --> calcolo spo2 --> slido finestra --> inferenza
  while (true) {

    while (1) { 

      int begin = millis();
      int32_t acc_valid = 0;
      int32_t count_valid = 0;
      // Serial.print("comincio\n");
      while (millis() - begin < 800) {
        
        // acquisisco campione
    
        while (!particleSensor.available()) particleSensor.check(); // aspetta
    
        memmove(redBuffer, redBuffer + 1, (RAW_BUF_LEN - 1) * sizeof(redBuffer[0]));
        memmove(irBuffer , irBuffer  + 1, (RAW_BUF_LEN - 1) * sizeof(irBuffer[0]));
    
        redBuffer[RAW_BUF_LEN - 1] = particleSensor.getRed();
        irBuffer [RAW_BUF_LEN - 1] = particleSensor.getIR();
        particleSensor.nextSample();
    
        // calcolo l'spo2
        maxim_heart_rate_and_oxygen_saturation(irBuffer, RAW_BUF_LEN, redBuffer, &spo2, &validSPO2, &heartRate, &validHeartRate);
        bool cond = spo2 < spo2_window[0] - 5 || spo2 > spo2_window[0] + 5; // non usata
        if (!validSPO2 || spo2 < 95) {
          // Serial.print(" balzo = ");
          // Serial.print(spo2);
          continue; // salta frame se corrotto
        }

        acc_valid = acc_valid + spo2;
        count_valid++;
      }

      // aggiorno finestra con l'helper che la fa slide-are
      if (acc_valid != 0 && countInvSpo2Med == 0) { // se il valore corrente e' valido e quelli prima erano validi

        // Serial.print("valori ok, count_valid =");
        // Serial.print(count_valid);
        // Serial.print("\n");
        updateSpo2Window((float)acc_valid/count_valid);
        break;

      } else if (acc_valid != 0 && countInvSpo2Med > 0) { // se il valore corrente e' il primo valido ma quelli prima erano invalidi
        // Serial.print("valori ok, count_valid =");
        // Serial.print(count_valid);
        // Serial.print("\n");

        updateSpo2Window((float)acc_valid/count_valid);

        float start = spo2_window[0]; // valore valido di partenza
        float end = spo2_window[countInvSpo2Med]; // valore valido di arrivo
        float step = (end - start) / (float)countInvSpo2Med; // pendenza

        for (int x = 1; x < countInvSpo2Med; ++x) { // riempi i soli campioni invalidi
          spo2_window[x] = start + step * x; // interpolazione lineare
        }
        // for (int x = 1; x < countInvSpo2Med; x++) {
        //   spo2_window[x] = spo2_window[0] + (float)((float)spo2_window[countInvSpo2Med]/(float)(countInvSpo2Med)) * (float)x;
        // }
        // Serial.print("Ho interpolato e ricominciato");
        countInvSpo2Med = 0;
        break;
      } else { // valore invalido e quelli precedenti erano invalidi
        // Serial.print("valore invalido, spo2= ");
        // Serial.print(spo2);
        // Serial.print(" acc_valid = ");
        // Serial.print(acc_valid);
        // Serial.print(" countInv = ");
        // Serial.print(countInvSpo2Med);
        // Serial.print("\n");
        countInvSpo2Med++;
      }
    }

    /* .4) normalizza e inferisci */
    uint32_t t0_prep = micros();

    float spo2_norm[N_SAMPLES];
    // Serial.print("\n");
    for (uint8_t i = 0; i < N_SAMPLES; ++i) {
      spo2_norm[i] = (spo2_window[i] - GLOBAL_MEAN_SPO2) / GLOBAL_STD_SPO2;
      // Serial.print(spo2_window[i]);
      // Serial.print(" ");
    }
    // Serial.print("\n");

    uint32_t t1_prep = micros();

    uint32_t input_shape[] = {N_SAMPLES, 1};
    float* logit_out;
    uint32_t t0_inf = micros();

    predict(spo2_norm, input_shape, 2, &logit_out);
    uint32_t t1_inf = micros();

    // 5) statistiche 
    loop_counter++; // per debugging
    prep_acc_us  += (t1_prep - t0_prep);
    infer_acc_us += (t1_inf  - t0_inf);

    float probability = sigmoidf(logit_out[0]);
    float avg_prep_ms = (prep_acc_us  / (float)loop_counter) / 1000.0f;
    float avg_infer_ms = (infer_acc_us / (float)loop_counter) / 1000.0f;

    Serial.print(F("#"));
    Serial.print(loop_counter);
    Serial.print(F("  SpO2="));
    Serial.print(spo2_window[0]);
    Serial.print(F("%  Prob="));
    Serial.print(probability, 4);
    Serial.print(F("  ⌛prep_avg="));
    Serial.print(avg_prep_ms, 2);
    Serial.print(F(" ms  ⌛infer_avg="));
    Serial.print(avg_infer_ms, 2);
    Serial.println(F(" ms"));

    // Serial.print(spo2_window[0], 1);
    // Serial.print(',');
    // Serial.print(probability, 4);
    // Serial.print(','); 
    // Serial.println(millis()); 
  }
}