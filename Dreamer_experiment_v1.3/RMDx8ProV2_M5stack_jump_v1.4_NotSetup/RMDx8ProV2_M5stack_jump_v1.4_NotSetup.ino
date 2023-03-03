/*
  RMD X8 Pro V2(Type1)用プログラム
  ***回転方向の符号注意***
  時計回り:マイナス（RMDX8ProV1:GYMESとは正負が異なる（20221124時点））

  使用するライブラリ（RMDX8ProV2_M5）も異なる
*/

#include <M5Stack.h>
#include <mcp_can_m5.h>
#include <SPI.h>
#include <RMDX8ProV2_M5.h>

#define BAUDRATE 115200 //シリアル通信がボトルネックにならないよう，速めに設定しておく


const uint16_t MOTOR_ADDRESS_R = 0x141; //0x140 + ID(1~32)
const int SPI_CS_PIN = 12; // M5stack

//unsigned long timer[3]; // 0:初期時間，1:現在の時間
unsigned long timer[5]; // 0:初期時間，1:現在の時間
float ang_R[2]; // 0:初期角度，1:現在の角度 [degree]
float ang_vel_R = 0; // 角速度 [degree/s]

bool exit_tf = false;

float positioning_torq_R = 0; // 位置決め目標トルク [Nm]
float tgt_torq_R = 0; // 目標トルク

float inf_torq_R = 0; // 推論トルク
int rl_signal = 1; // 0:送受信, 1:受信のみ, 2:終了コマンド

int16_t tgt_cur_R; // 目標電流 [-]

int8_t mode_R = 0; // 0:振り戻し, 1:振り出し

int8_t motor_state = 0; // トルク制限や角度制限時に，PC側に何が起きたかを伝える

// トルク制限
float backward_max_torq = -5; // 振り戻し最大トルク
float forward_max_torq = 5; // 振り出し最大トルク

// 角度制限
float backward_max_ang = -140; // 振り戻し最大角度
float forward_max_ang = 10; // 振り出し最大角度

MCP_CAN_M5 CAN0(SPI_CS_PIN); // set CS PIN
RMDX8ProV2_M5 rmd_R(CAN0, MOTOR_ADDRESS_R);

void setup()
{
  M5.begin();

  delay(2000);

  M5.Power.begin();
  Serial.begin(BAUDRATE);
  while (!Serial);

  init_can();

  rmd_R.canSetup();
  rmd_R.clearState(); // 初期化
  rmd_R.writePID(40, 40, 50, 40, 20, 20); // PID gainの設定 angleKp, angleKi, speedKp, speedKi, iqKp, iqKi
  delay(100);

  Serial.println("Not Setting");
  delay(100);

  timer[0] = millis();

  // ゼロ調整 torque --> 6.0[Nm]
  // while (millis() < timer[0] + 4000)
  // {
  //   if (positioning_torq_R < 6.0)
  //   {
  //     positioning_torq_R = positioning_torq_R + 0.2;
  //   }

  //   tgt_cur_R = Torque_to_Current(positioning_torq_R);

  //   rmd_R.writeCurrent(tgt_cur_R);
  //   delay(1);
  //   rmd_R.readPosition();

  //   ang_R[0] = rmd_R.present_angle;
    
  //   Serial.print("torque_R:");
  //   Serial.print(positioning_torq_R);
  //   Serial.print(", time:");
  //   Serial.print(millis() - timer[0]);
  //   Serial.print(", ang_R:");
  //   Serial.println(ang_R[0]);

  //   delay(50);
  // }

  // ang_R[0] = ang_R[0];
  // ang_R[1] = rmd_R.present_angle - ang_R[0];

  // Serial.print("ang_R[0]:");
  // Serial.print(ang_R[0]);
  // Serial.print(",ang_R[1]:");
  // Serial.println(ang_R[1]);

  // delay(100);

  // // torque --> 0[Nm]
  // timer[0] = millis();
  // while (millis() < timer[0] + 4000)
  // {
  //   if (positioning_torq_R > 0.0)
  //   {
  //     positioning_torq_R = positioning_torq_R - 0.2;
  //     Serial.print("torque R:");
  //     Serial.println(positioning_torq_R);
  //   }
    
  //   tgt_cur_R = Torque_to_Current(positioning_torq_R);
    
  //   rmd_R.writeCurrent(tgt_cur_R);
    
  //   delay(50);
  // }
  rmd_R.writeCurrent(0);

  delay(100);
  Serial.println();
  Serial.println("start_m5");
  
  timer[0] = millis();
}

void loop()
{
  while (exit_tf == false)
  {
    DJRead_m5();
    
    timer[1] = millis() - timer[0]; // 現在の時刻 [ms]

    tgt_torq_R = inf_torq_R;

    // トルク制限
    if (tgt_torq_R < backward_max_torq)
    {
      tgt_torq_R = backward_max_torq;
      motor_state = 1;
    }
    else if (tgt_torq_R > forward_max_torq)
    {
      tgt_torq_R = forward_max_torq;
      motor_state = 2;
    }

    // 角度制限
    if (ang_R[1] <= backward_max_ang)
    {
      // Serial.println("Right_backward_max_ang_over");
      // tgt_torq_R = 0;
      motor_state = 3;
      // exit_tf = true;
    }
    else if (ang_R[1] >= forward_max_ang)
    {
      // Serial.println("Right_forward_max_ang_over");
      // tgt_torq_R = 0;
      motor_state = 4;
      // exit_tf = true;
    }

    // Action
    tgt_cur_R = Torque_to_Current(tgt_torq_R);
    
    rmd_R.writeCurrent(tgt_cur_R);
    // rmd_R.writeCurrent(0);

    delay(1);

    // Sensing
    rmd_R.readPosition(); // 現在の位置(角度)を取得

    ang_R[1] = rmd_R.present_angle - ang_R[0]; // モータ角度 [deg]
    ang_vel_R = rmd_R.present_vel; // モータ角速度 [deg/s]

    // Print
    if (rl_signal == 0)
    {
      serial_disp();
    }
    
    // 終了判定
    if (rl_signal == 2)
    {
      tgt_torq_R = 0;
      tgt_cur_R = 0;
      exit_tf = true;
    }
  }

  // loop out
  rmd_R.writeCurrent(0);
  rmd_R.clearState(); // 初期化

  while (true)
  {
    delay(1000);
  }
}

void DJRead_m5() {
  while (Serial.available() == 0)
  {
//     delayMicroseconds(500);
  }
  String str_data0 = Serial.readStringUntil('\0');
  String str_data2 = Serial.readStringUntil('\2');
  inf_torq_R = str_data0.toFloat();
  rl_signal = str_data2.toFloat();
}

int16_t Torque_to_Current(float torque) {
  /*
    --換算式--
    ・電流I-電流指令値I_d：I = a*I_d + b*I_d + c
    ・トルクtau-電流指令値I_d：tau = d*I_d + e   ==>   I_d = (tau - e) / d
    I_d, I, tau:電流指令値，実際に流れた電流，実際に発揮されたトルク
    a, b, c, d, e:係数(モータの種類による)
    <RMD-X8ProV2係数（太田卒論より参照）>
    a:7.49*10^-6
    b:-1.6*10^-3
    c:0.189
    d:3.6*10^-2
    e:9.17*10^-2
  */

  int16_t current;
  current = (torque - 9.17*0.01) / (3.6*0.01);
  
  return current;
}

void serial_disp() {
  Serial.print(timer[1]);
  Serial.print(",");
  Serial.print(tgt_torq_R);
  Serial.print(",");
  Serial.print(ang_R[1]);
  Serial.print(",");
  Serial.print(ang_vel_R);
  Serial.print(",");
  Serial.print(rmd_R.present_current);
  Serial.print(",");
  Serial.print(rmd_R.temperature);
  Serial.print(",");
  Serial.println(motor_state);
}

void init_can()
{
  // Initialize MCP2515 running at 16MHz with a baudrate of 500kb/s and the masks and filters disabled.
  if (CAN0.begin(MCP_ANY, CAN_1000KBPS, MCP_8MHZ) == CAN_OK)
  {
    Serial.println("MCP2515 Initialized Successfully!");
  }
  else
  {
    Serial.println("Error Initializing MCP2515...");
  }

  CAN0.setMode(MCP_NORMAL); // Change to normal mode to allow messages to be transmitted
}
