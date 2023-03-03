#include <Wire.h>
#include <Adafruit_MCP4725.h>
#include <AS5X47.h>

#define BAUDRATE 115200 //シリアル通信がボトルネックにならないよう，速めに設定しておく
#define neutral_threshold 0 // ニュートラル判定の閾値
#define PI 3.141592653589793 // 円周率

// Pin number
#define DIGITAL_OUT_PULL 25    // 電磁弁操作1
#define DIGITAL_OUT_PUSH 27    // 電磁弁操作2

#define MARKER_CONTACT 0    // スライダ圧力センサ

#define FORCE_SENSOR_1 4    // 力センサ1
#define FORCE_SENSOR_2 5    // 力センサ2
#define FORCE_SENSOR_3 6    // 力センサ3
#define FORCE_SENSOR_4 7    // 力センサ4
#define FORCE_SENSOR_5 8    // 力センサ5

#define LINEAR_ENCODER_PIN_A 19    // リニアエンコーダ_A相
#define LINEAR_ENCODER_PIN_B 18    // リニアエンコーダ_B相

#define AS5047P_CHIP_SELECT_PORT 53 // define the chip select port.

#define HSC_SYNC_PIN 13 // HSC

Adafruit_MCP4725 dac; // レギュレータ
AS5X47 as5047p_center(AS5047P_CHIP_SELECT_PORT); // ロータリーエンコーダ

bool sync_hsc = false;
bool first_loop = true;

//unsigned long timer[3]; // 0:初期時間, 1:現在の時間, 2:差分
unsigned long timer[5]; // 0:初期時間, 1:現在の時間, 2:差分, 3:一つ前のループの時間

int16_t tgt_pressure = 0; // レギュレータ圧力 0~4095(2^12)
float tgt_valve = 0; // 空圧アクチュエータ -1~1

int16_t inf_pressure = 0; // 推論用レギュレータ圧力 0~4095(2^12)
float inf_valve = 0; // 推論用空圧アクチュエータ -1~1
int rl_signal = 1; // 0:送受信, 1:受信のみ, 2:終了コマンド

bool exit_tf = false;

// Sensor
const int8_t ENCODER_TABLE[] = {0,-1,1,0,1,0,0,-1,-1,0,0,1,0,1,-1,0};
volatile bool StatePinA = 1;
volatile bool StatePinB = 1;
volatile uint8_t State = 0;
volatile long Count = 0;

float difference = 0; // 現在の距離と初期状態の距離の差分[mm]

float angle_boom = 0; // ブーム回転角度[deg]
float pre_angle_boom = 0;
float velocity_boom = 0;

int16_t p_marker = 0; // 圧力センサ

int16_t force_sensor_1 = 0; // 力センサ1 [kg]
int16_t force_sensor_2 = 0; // 力センサ2
int16_t force_sensor_3 = 0; // 力センサ3
int16_t force_sensor_4 = 0; // 力センサ4
int16_t force_sensor_5 = 0; // 力センサ5

void setup() {
//  Wire.begin();
  digitalWrite(SDA, 1);
  digitalWrite(SCL, 1);

  pinMode(LINEAR_ENCODER_PIN_A, INPUT_PULLUP);
  pinMode(LINEAR_ENCODER_PIN_B, INPUT_PULLUP);
  pinMode(DIGITAL_OUT_PULL, OUTPUT);
  pinMode(DIGITAL_OUT_PUSH, OUTPUT);

  pinMode(HSC_SYNC_PIN, OUTPUT);
  digitalWrite(HSC_SYNC_PIN, LOW);

  Serial.begin(BAUDRATE);
  while (!Serial);

  attachInterrupt(digitalPinToInterrupt(LINEAR_ENCODER_PIN_A), ChangePinAB, CHANGE);
  attachInterrupt(digitalPinToInterrupt(LINEAR_ENCODER_PIN_B), ChangePinAB, CHANGE);
  
  // For Adafruit MCP4725A1 the address is 0x62 (default) or 0x63 (ADDR pin tied to VCC)
  // For MCP4725A0 the address is 0x60 or 0x61
  // For MCP4725A2 the address is 0x64 or 0x65

  dac.begin(0x60);
  delay(500);

//  dac.setVoltage(1024, false);
//  delay(500);
//  setPush(DIGITAL_OUT_PUSH, DIGITAL_OUT_PULL);
//  delay(500);

  angle_boom = as5047p_center.readAngle(); // 速度計算のために開始前の角度を取得

  delay(1000);

  Serial.println("start_arduino");

  timer[0] = millis();
  timer[1] = millis() - timer[0]; // 現在の時刻 [ms]
}

void loop() {
  while (exit_tf == false)
  {
    DJRead_arduino();
    
    timer[3] = timer[1]; // 一つ前のループの時間を格納
    timer[1] = millis() - timer[0]; // 現在の時刻 [ms]
    
    if (sync_hsc == false && rl_signal == 0)
    {
      digitalWrite(HSC_SYNC_PIN, HIGH);
      sync_hsc = true;
    }

    tgt_valve = inf_valve;
    tgt_pressure = inf_pressure;

    if (tgt_pressure > 4095)
    {
      tgt_pressure = 4095;
    }
    else if (tgt_pressure < 0)
    {
      tgt_pressure = 0;
    }

    // Action
    dac.setVoltage(tgt_pressure, false);

    if (tgt_valve <= neutral_threshold)
    {
      setPull(DIGITAL_OUT_PUSH, DIGITAL_OUT_PULL);
    }
    else
    {
      setNeutral(DIGITAL_OUT_PUSH, DIGITAL_OUT_PULL);
    }

    delay(1);

    // Sensing
    p_marker = analogRead(MARKER_CONTACT);

    force_sensor_1 = analogRead(FORCE_SENSOR_1);
    force_sensor_2 = analogRead(FORCE_SENSOR_2);
    force_sensor_3 = analogRead(FORCE_SENSOR_3);
    force_sensor_4 = analogRead(FORCE_SENSOR_4);
    force_sensor_5 = analogRead(FORCE_SENSOR_5);

    // 出力値をkg単位へと変換
    force_sensor_1 = force_sensor_1*0.0546 - 0.8485;
    if(force_sensor_1 < 0) force_sensor_1 = 0;

    force_sensor_2 = force_sensor_2*0.0531 - 0.3410;
    if(force_sensor_2 < 0) force_sensor_2 = 0;

    force_sensor_3 = force_sensor_3*0.0514 - 0.9199;
    if(force_sensor_3 < 0) force_sensor_3 = 0;

    force_sensor_4 = force_sensor_4*0.0535 - 1.7801;
    if(force_sensor_4 < 0) force_sensor_4 = 0;

    force_sensor_5 = force_sensor_5*0.0490 - 0.3870;
    if(force_sensor_5 < 0) force_sensor_5 = 0;

    noInterrupts();
    difference = Count * 0.1 / 4;
    interrupts();

    pre_angle_boom = angle_boom; // 一つ前のループの角度を格納
    angle_boom = as5047p_center.readAngle(); // 現在のブームの角度を取得
    
    velocity_boom = -(angle_boom - pre_angle_boom) / ((timer[1] - timer[3]) * 0.001); // [deg/s](マイナスは，前進方向への回転でブーム角度が減少するため)
    velocity_boom = velocity_boom * (1.13 * PI / 180.0); // [m/s]
    if (first_loop == true) // 例外処理．初回ループでは速度を0とする．
    {
      velocity_boom = 0;
      first_loop = false;
    }

    // Print
    if (rl_signal == 0)
    {
      serial_disp();
    }
    
    // 終了判定
    if (rl_signal == 2)
    {
      exit_tf = true;
    }
  }

  // loop out
  setNeutral(DIGITAL_OUT_PUSH, DIGITAL_OUT_PULL);
  dac.setVoltage(0, false);

  digitalWrite(HSC_SYNC_PIN, LOW);

  while (true)
  {
    delay(1000);
  }
}

void DJRead_arduino() {
  while (Serial.available() == 0)
  {
    // delayMicroseconds(500);
  }
  String str_data0 = Serial.readStringUntil('\0');
  String str_data1 = Serial.readStringUntil('\1');
  String str_data2 = Serial.readStringUntil('\2');
  inf_valve = str_data0.toFloat();
  inf_pressure = str_data1.toFloat();
  rl_signal = str_data2.toFloat();
}

void setPush(int push_pin, int pull_pin) {
  digitalWrite(push_pin, HIGH);
  digitalWrite(pull_pin, LOW);
}

void setPull(int push_pin, int pull_pin) {
  digitalWrite(push_pin, LOW);
  digitalWrite(pull_pin, HIGH);
}

void setNeutral(int push_pin, int pull_pin) {
  digitalWrite(push_pin, LOW);
  digitalWrite(pull_pin, LOW);
}

void ChangePinAB() {
  StatePinA = PIND & 0b00000100;
  StatePinB = PIND & 0b00001000;
  State = (State<<1) + StatePinA;
  State = (State<<1) + StatePinB;
  State = State & 0b00001111;
  Count += ENCODER_TABLE[State];
}

void serial_disp() {
  Serial.print(timer[1]);
  Serial.print(",");
  Serial.print(tgt_valve);
  Serial.print(","); 
  Serial.print(tgt_pressure);
  Serial.print(",");
  Serial.print(difference);
  Serial.print(",");
  Serial.print(angle_boom);
  Serial.print(",");
  Serial.print(velocity_boom, 4); // 小数点以下の桁数を4とする．
  Serial.print(",");
  Serial.print(p_marker);
  Serial.print(",");
  Serial.print(force_sensor_1);
  Serial.print(",");
  Serial.print(force_sensor_2);
  Serial.print(",");
  Serial.print(force_sensor_3);
  Serial.print(",");
  Serial.print(force_sensor_4);
  Serial.print(",");
  Serial.println(force_sensor_5);
}
