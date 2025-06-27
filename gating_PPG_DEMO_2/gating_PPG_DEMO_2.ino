#include <Arduino_BHY2.h>
#include <Nicla_System.h>

const float   COUNTS_PER_G     = 4096.0f;
const float   EXTRA_MARGIN     = 0.05f;    
const int     CALIB_TIME_MS    = 5000;  
const uint32_t MOTION_INTERVAL = 100;      
const uint32_t PPG_INTERVAL    = 1000;   

SensorXYZ accel(SENSOR_ID_ACC);

float   lowThresh   = 0.0f;
float   highThresh  = 0.0f;
bool    lastMoving  = false;

uint32_t lastMotion = 0;
uint32_t lastPPG    = 0;

void setup() {
  // Serial e breve attesa per il Monitor 
  Serial.begin(115200);
  uint32_t tStart = millis();
  while (!Serial && millis() - tStart < 2000);
  Serial.println(F("Serial OK, demo 10 Hz IMU + 1 Hz PPG fittizio"));

  // Inizializza IMU e LED 
  nicla::begin();
  BHY2.begin(NICLA_I2C);
  accel.begin(10, 0);
  nicla::leds.begin();
  nicla::leds.setColor(green);

  // Calibrazione IMU (min/max) per definire la fascia di quiete
  Serial.println(F("Calibrazione IMU (5 s)…"));
  float minMag = 10.0f, maxMag = 0.0f;
  uint32_t tCalib = millis();
  while (millis() - tCalib < CALIB_TIME_MS) {
    BHY2.update();
    if (accel.dataAvailable()) {
      float x = accel.x() / COUNTS_PER_G;
      float y = accel.y() / COUNTS_PER_G;
      float z = accel.z() / COUNTS_PER_G;
      float mag = sqrt(x*x + y*y + z*z);
      minMag = min(minMag, mag);
      maxMag = max(maxMag, mag);
    }
  }
  float center   = (minMag + maxMag) * 0.5f;
  float halfBand = (maxMag - minMag) * 0.5f;
  lowThresh  = center - (halfBand + EXTRA_MARGIN);
  highThresh = center + (halfBand + EXTRA_MARGIN);

  Serial.print(F("Quiet zone: "));
    Serial.print(lowThresh, 3);
    Serial.print(F(" – "));
    Serial.print(highThresh, 3);
    Serial.println(F(" g"));

  // Inizializza timer per loop
  lastMotion = millis();
  lastPPG    = millis();
  randomSeed(micros());
}

void loop() {
  uint32_t now = millis();

  // Gating movimento a 10 Hz
  if (now - lastMotion >= MOTION_INTERVAL) {
    lastMotion = now;
    BHY2.update();

    if (accel.dataAvailable()) {
      float x = accel.x() / COUNTS_PER_G;
      float y = accel.y() / COUNTS_PER_G;
      float z = accel.z() / COUNTS_PER_G;
      float mag = sqrt(x*x + y*y + z*z);
      bool moving = (mag < lowThresh) || (mag > highThresh);

      if (moving != lastMoving) {
        lastMoving = moving;
        nicla::leds.setColor(moving ? red : green);
      }
    }
  }

  // PPG fittizio + stampa a 1 Hz
  if (now - lastPPG >= PPG_INTERVAL) {
    lastPPG = now;

    // genera valori PPG casuali
    uint32_t red = random(20000, 30000);
    uint32_t ir  = random(20000, 30000);

    // se fermo, stampa RED/IR; altrimenti solo "MOVIMENTO"
    if (!lastMoving) {
      Serial.print(now / 1000.0f, 3);
      Serial.print(F(" s  RED_FIT=")); Serial.print(red);
      Serial.print(F("  IR_FIT="));    Serial.println(ir);
    } else {
      Serial.print(now / 1000.0f, 3);
      Serial.println(F(" s  MOVIMENTO"));
    }
  }
}
