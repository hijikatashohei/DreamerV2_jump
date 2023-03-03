import os
from time import time
from base64 import b64encode
import math
import numpy as np
import torch
import matplotlib.pyplot as plt

import serial
import time
import csv
from ximea import xiapi
import datetime
import re
import rtmidi2
from PIL import Image
# import threading

from utils import define_action_to_be_selected


class Environment_410:
    def __init__(self, r_hsc, data_ex, action_discrete, bottom_height, episode_end_steps=500):

        self.r_hsc = r_hsc
        self.data_ex = data_ex
        self.action_discrete = action_discrete # True or False
        self.episode_end_steps = episode_end_steps
        self.episode = 1

        pressure_action_discrete = np.array([0.0, 0.25, 0.5]) # 圧力[MPa]
        motor_action_discrete = np.array([-1.0, 0.0, 2.0]) # RMDX8ProV2:右脚 振り出し:+, 振り戻し:-
        
        self.action_output = define_action_to_be_selected(pressure_action_discrete, motor_action_discrete)
        print("action_discrete", self.action_discrete)

        self.episode_steps = 0
        self.total_steps = 0

        # その他パラメータ．
        self.n = 0
        self.bottom_height = bottom_height
        self.pre_x = -self.bottom_height
        self.pre_angle = 0

        self.done = False
        self.end_sign = False

    def step(self, _action:np.array, step_timer:float, train_mode:bool=True):
        '''
            env.step(action) -> next_obs, reward, done, info
            discrete:
                action : one-hot vector, dim=[action_space], dtype=numpy int
                ex:[0 0 0 0 1 0 0 0 0 0]
            continuous:
                action : vector, dim=[action_space], dtype=numpy float
                ex:[0.0, -5.5]
        '''
        assert self.done == False, "done is True.should be False"

        if self.episode_steps == 0:
            self.loop_time_0 = time.perf_counter()

        self.episode_steps += 1
        
        if train_mode:
            self.total_steps += 1

        self.loop_time = time.perf_counter() - self.loop_time_0
        # print("a")

        # action決定
        if self.action_discrete:
            assert len(_action.shape) == 1, "_action.shape is not 1"
            action_number = np.argmax(_action, axis=0).item()
            # action[0]:空圧 pull:a_0<=0, neutral:0<a_0, action[1]:レギュレータ 0~4095

            if self.action_output[action_number][0] == 0:
                action = [1, 0, self.action_output[action_number][1]]
                # print(action_number)
                # print(action)
            
            else:
                action = [-1, int(self.action_output[action_number][0] / 0.6 * 4095), self.action_output[action_number][1]]

        else: # continuous control
            raise NotImplementedError
            # action = _action
            # action = [_action[0], _action[1]]

        print("action_number", action_number)
        print("action", action)
        
        # print("b")
        loop_interval1 = (time.perf_counter() - self.loop_time_0) - self.loop_time

        self.data_ex.serial_send(action) # action等をマイコンに送信

        # print("c")
        loop_interval2 = (time.perf_counter() - self.loop_time_0) - self.loop_time

        self.data_ex.catch_nanoKON(delay=False)

        if self.data_ex.nanokon == 42:
            self.done = True
            self.end_sign = True

            print("pushed 42")

        elif self.data_ex.nanokon == 43:
            self.done = True

            print("pushed 43")

        elif self.data_ex.nanokon == 44:
            self.done = True

            print("pushed 44")

        if self.episode_steps == self.episode_end_steps:
            self.done = True

        # M5Stackからのデータを受信
        receive_m5 = self.data_ex.m5stack_read()
        print("m5stack : ", receive_m5)

        # Arduinoからのデータ受信
        receive_arduino = self.data_ex.arduino_read()
        print("arduino : ", receive_arduino)

        # print("e")
        loop_interval3 = (time.perf_counter() - self.loop_time_0) - self.loop_time

        next_state = self.r_hsc.get_state() # 次の状態を取得

        # print("d")
        loop_interval4 = (time.perf_counter() - self.loop_time_0) - self.loop_time

        # receive_arduino = [0]:timer, [1]:tgt_valve, [2]:tgt_pressure, [3]:difference, [4]:angle_boom, [5]:velocity_boom, [6]:p_slider, [7]:force_sensor_1, [8]:force_sensor_2, [9]:force_sensor_3, [10]:force_sensor_4,  [11]:force_sensor_5
        # receive_m5 = [0]:timer, [1]:tgt_torq_R, [2]:ang_R, [3]:ang_vel_R, [4]:present_current, [5]:motor_state
        reward = self.return2reward(receive_arduino[3], receive_arduino[5], receive_arduino[11]) # このstepでのrewardを取得

        # print("f")
        loop_interval5 = (time.perf_counter() - self.loop_time_0) - self.loop_time

        if train_mode: # データを保存する．
            self.data_ex.csv_write(receive_arduino, receive_m5, self.episode, self.total_steps, reward, step_timer, train_mode, self.done)

        else:
            self.data_ex.csv_write(receive_arduino, receive_m5, self.episode, self.total_steps, reward, step_timer, train_mode, self.done)

        loop_interval = (time.perf_counter() - self.loop_time_0) - self.loop_time

        print('steps: {0:.5f}, {1}, {2}, {3}, {4}, {5}, {6:.5f}'.format(step_timer, self.total_steps, action[0], action[1], reward, self.episode_steps, loop_interval))

        # print("g")
        print('time_interval: {0:.5f}, {1:.5f}, {2:.5f}, {3:.5f}, {4:.5f}, {5:.5f}'.format(loop_interval, loop_interval1, loop_interval2, loop_interval3, loop_interval4, loop_interval5))

        return next_state, reward, self.done, self.end_sign


    # エピソードが終了した場合には，環境をリセットする．
    def preparation_for_next_episode(self, episode):

        self.episode = episode

        # 次のエピソードに向けて，各変数の値を初期化
        self.episode_steps = 0
        self.n = 0
        self.pre_x = self.bottom_height
        self.pre_angle = 0

        state = None  # 次の状態はないので、Noneを格納

        if not self.end_sign:
            print("Please select Restart(45) or End(42)")
            while True:
                self.data_ex.catch_nanoKON()

                if self.data_ex.nanokon == 45: # 次のepisodeに向けての準備
                    print("Arduino and M5Stack are Setting Now")
                    DEVICE = "COM20"
                    BAUDRATE = 115200
                    TIMEOUT = 1  # [s]
                    self.data_ex.ser_mega = serial.Serial(DEVICE, BAUDRATE, timeout=TIMEOUT)

                    # M5stack
                    # DEVICE = "COM13"
                    DEVICE = "COM25"
                    BAUDRATE = 115200
                    TIMEOUT = 1  # [s]
                    self.data_ex.ser_m5 = serial.Serial(DEVICE, BAUDRATE, timeout=TIMEOUT)

                    # シリアル通信オープン時に，M5stackをリセットする（毎回ボタンを押す必要をなくす）
                    self.data_ex.ser_m5.setDTR(False)
                    time.sleep(0.1)
                    self.data_ex.ser_m5.setRTS(False)
                    self.data_ex.ser_m5.rtscts = False

                    self.data_ex.read_start_sign()
                    # 環境を初期化する．= 今回はロボットの状態を人が戻す．
                    action = [1, 0, 0]
                    self.data_ex.nanoKON_send(action)

                    print("Please Push Start(41) or Back(43)")
                    while True:
                        self.data_ex.catch_nanoKON()

                        if self.data_ex.nanokon == 41: # 次のepisodeを開始
                            break

                        if self.data_ex.nanokon == 43: # Restart(45) or End(42)に戻る
                            print("Pushed Back(43)")
                            print("Please select Restart(45) or End(42)")

                            self.data_ex.serial_loop_out() # アクチュエータの実行を終了
                            self.data_ex.ser_mega.close()
                            self.data_ex.ser_m5.close()
                            break

                    if self.data_ex.nanokon == 41: # 次のepisodeを開始
                        break

                elif self.data_ex.nanokon == 42:
                    print("Selected End(42)")
                    self.end_sign = True
                    break

        state = self.r_hsc.get_state()
        self.done = False

        return state, self.end_sign

    def return2reward(self, distance, velocity_boom, force_stopper):
        
        # 9月28日
        x = float(distance) - self.bottom_height # Arduino側は高さを取るだけで，背伸び高さをゼロとする計算はPC側で実施．
        y = x - self.pre_x

        if(-5 <= y <= 5):
            reward = -0.1
        else:
            reward = 0

        self.pre_x = x

        if(15 <= x < 20):
            reward = reward + (0.16*x - 2.2)
        elif(20 <= x < 25):
            reward = reward + (-0.16*x + 4.2)
        elif(25 <= x):
            reward = reward - 0.1

        return reward


class DataExchange:
    def __init__(self, directory, exp_info):
        self.directory = directory
        self.exp_info = exp_info

        self.nanokon = 0
        self.sendflag = False

        midi_in = rtmidi2.MidiIn()
        print(midi_in.ports)

        device_name = "nanoKONTROL2"
        try:
            index = midi_in.ports_matching(device_name+"*")[0]
            self.input_port = midi_in.open_port(index)
        except IndexError:
            raise(IOError("Input port not found."))

        # 管理フォルダを作成
        if os.path.isdir(self.directory) == False:
            os.makedirs(self.directory)
            os.chdir(self.directory)
            os.makedirs('Train')
            os.makedirs('Evaluate')
            os.makedirs('Loss')
            os.makedirs('Prediction')
            os.makedirs('Save_Buffer')
            os.makedirs('Train_Weights')
            os.chdir('Train')
        else:
            os.chdir(self.directory)
            os.chdir('Train')

        date = datetime.datetime.today()
        a0 = date.year
        b0 = date.month
        c0 = date.day
        a=str(a0).zfill(4)
        b=str(b0).zfill(2)
        c=str(c0).zfill(2)

        if exp_info == 'run_seed_episode' or exp_info == 'train':
            dir = os.getcwd()
            files = os.listdir(dir)
            count = 1
            for file in files:
                index = re.search('.csv', file)
                if index:
                    count = count + 1
            # count = int(count/2)
            file_count_0 = count
            file_count = str(file_count_0).zfill(2)
            # ログデータのヘッダー
            header1 = ['time_arduino','tgt_valve','tgt_pressure','difference','angle_boom','velocity_boom','p_marker','force_sensor_1','force_sensor_2','force_sensor_3','force_sensor_4','force_sensor_5','time_m5','tgt_torq_R','ang_R','ang_vel_R','present_current','temperature','motor_state','episode','steps','reward','step_timer','done']

            self.f1 = open(str(a) + str(b) + str(c) + '301' + str(file_count) + '_' + exp_info +'.csv','a',newline="")
            self.writer1 = csv.writer(self.f1)
            self.writer1.writerow(header1)

        os.chdir('../')
        os.chdir('Evaluate')

        if exp_info == 'train' or exp_info == 'test':
            dir = os.getcwd()
            files = os.listdir(dir)
            count = 1
            for file in files:
                index = re.search('.csv', file)
                if index:
                    count = count + 1
            file_count_0 = count
            file_count = str(file_count_0).zfill(2)
            # ログデータのヘッダー
            header2 = ['time_arduino','tgt_valve','tgt_pressure','difference','angle_boom','velocity_boom','p_marker','force_sensor_1','force_sensor_2','force_sensor_3','force_sensor_4','force_sensor_5','timer_m5','tgt_torq_R','ang_R','ang_vel_R','present_current','temperature','motor_state','episode','steps','reward','step_timer','done']

            self.f2 = open(str(a) + str(b) + str(c) + '304' + str(file_count) + '_' + exp_info + '.csv','a',newline="")
            self.writer2 = csv.writer(self.f2)
            self.writer2.writerow(header2)

        os.chdir('../')
        os.chdir('Loss')
        
        if exp_info == 'train':
            dir = os.getcwd()
            files = os.listdir(dir)
            count = 1
            for file in files:
                index = re.search('.csv', file)
                if index:
                    count = count + 1
            file_count_0 = count
            file_count = str(file_count_0).zfill(2)
            # ログデータのヘッダー
            header3 = ['episode','imagine_iteration','update_step','model_loss','kl_loss','obs_loss','reward_loss','discount_loss','value_loss','action_loss']
            self.f3 = open(str(a) + str(b) + str(c) + '306' + str(file_count) + '_' + exp_info + '.csv','a',newline="")
            self.writer3 = csv.writer(self.f3)
            self.writer3.writerow(header3)

        os.chdir('../')
        os.chdir('../')

        #Serial通信
        # Arduino
        DEVICE = "COM20"
        BAUDRATE = 115200
        TIMEOUT = 1  # [s]
        self.ser_mega = serial.Serial(DEVICE, BAUDRATE, timeout=TIMEOUT)

        # M5stack
        # DEVICE = "COM13"
        DEVICE = "COM25"
        BAUDRATE = 115200
        TIMEOUT = 1  # [s]
        self.ser_m5 = serial.Serial(DEVICE, BAUDRATE, timeout=TIMEOUT)

        # シリアル通信オープン時に，M5stackをリセットする（毎回ボタンを押す必要をなくす）
        self.ser_m5.setDTR(False)
        time.sleep(0.1)
        self.ser_m5.setRTS(False)
        self.ser_m5.rtscts = False

    def read_start_sign(self):
        while True:
            line = self.ser_mega.readline().rstrip().decode("ascii")
            receive = line.split(",")
            if receive[0] == "start_arduino":
                print("Arduino OK!")
                break

        while True:
            line = self.ser_m5.readline().rstrip().decode("ascii")
            receive = line.split(",")
            print(receive)
            if receive[0] == "start_m5":
                print("M5Stack OK!")
                break

    def serial_send(self, action): # rl_signal == 0
        letter_arduino = (str(action[0]) + "\0" + str(action[1]) + "\1" + str(0) + "\2").encode("ascii") # [pull, neutral], pressure
        letter_m5 = (str(action[2]) + "\0" + str(0) + "\2").encode("ascii") # torque_R
        
        self.ser_mega.write(letter_arduino)
        self.ser_m5.write(letter_m5)

    def nanoKON_send(self, action): # serial_sendとの違いは，マイコン側からのデータ受信を行わない(rl_signal=1)．，randomによる初期状態をつくるなど．
        letter_arduino = (str(action[0]) + "\0" + str(action[1]) + "\1" + str(1) + "\2").encode("ascii") # [pull, neutral], pressure
        letter_m5 = (str(action[2]) + "\0" + str(1) + "\2").encode("ascii") # torque_R

        self.ser_mega.write(letter_arduino)
        self.ser_m5.write(letter_m5)

    def serial_loop_out(self): # neutral, 圧力0, トルク0でloopを安全に終了．
        letter_arduino = (str(1) + "\0" + str(0) + "\1" + str(2) + "\2").encode("ascii") # [pull, neutral], pressure
        letter_m5 = (str(0) + "\0" + str(2) + "\2").encode("ascii") # torque_R

        self.ser_mega.write(letter_arduino)
        self.ser_m5.write(letter_m5)

    def arduino_read(self): # arduino側からデータ受信
        while self.ser_mega.in_waiting == 0:
            pass

        line = self.ser_mega.readline().rstrip().decode("ascii")
        receive = line.split(",")

        return receive

    def m5stack_read(self): # m5stack側からのデータを受信
        while self.ser_m5.in_waiting == 0:
            pass

        line = self.ser_m5.readline().rstrip().decode("ascii")
        receive = line.split(",")

        return receive

    def serial_reset(self):
        self.serial_loop_out() # アクチュエータの駆動を終了
        self.ser_mega.close() # マイコンとのシリアル通信を終了
        self.ser_m5.close()

    def csv_write(self, receive_arduino, receive_m5, episode, steps, reward, step_timer, train_mode, done): # csvへの書き込み(旧read_motor)
        receive_arduino.extend(receive_m5)

        receive_arduino.append(episode)
        receive_arduino.append(steps)
        receive_arduino.append(reward)
        receive_arduino.append(step_timer)
        receive_arduino.append(done)

        if train_mode:
            self.writer1.writerow(receive_arduino)
        else:
            self.writer2.writerow(receive_arduino)

    def csv_loss_write(self, episode, imag_iter, update_step, model_loss, kl_loss, obs_loss, rew_loss, disc_loss, val_loss, act_loss):
        loss_data = []

        loss_data.append(episode)
        loss_data.append(imag_iter)
        loss_data.append(update_step)
        loss_data.append(model_loss)
        loss_data.append(kl_loss)
        loss_data.append(obs_loss)
        loss_data.append(rew_loss)
        loss_data.append(disc_loss)
        loss_data.append(val_loss)
        loss_data.append(act_loss)

        self.writer3.writerow(loss_data)

    def catch_nanoKON(self, delay: bool=True): # コントローラからの指令を受け取る
        '''
            Threading : False
        '''
        if delay == True:
            time.sleep(0.01) # delayがないとRAMの値が増大する．必要に応じて使用．

        message = self.input_port.get_message()

        if message:
            self.sendflag = True
        
        if self.sendflag:
            self.nanokon = message[1]

            self.sendflag = False

    def reset_button_nanoKON(self):
        self.nanokon = 0

    def end_process(self): # csv保存．
        if self.exp_info == 'run_seed_episode' or self.exp_info == 'train':
            self.f1.close()
        if self.exp_info == 'train' or self.exp_info == 'test':
            self.f2.close()
        if self.exp_info == 'train':
            self.f3.close()

        time.sleep(1)
        self.ser_mega.close()
        self.ser_m5.close()


class Read2HSCv2:
    '''
        Timing : False
    '''
    # https://www.ximea.com/support/wiki/apis/python#Note
    def __init__(self, height, width):
        #HSC_setting
        self.cam = xiapi.Camera()
        self.cam.open_device()
        self.cam.set_exposure(1875)
        self.cam.set_downsampling_type("XI_SKIPPING")
        self.cam.set_downsampling("XI_DWN_2x2") # [512, 640]
        # self.cam.set_acq_timing_mode("XI_ACQ_TIMING_MODE_FRAME_RATE")
        # self.cam.set_framerate(framerate)
        self.img = xiapi.Image()
        self.cam.start_acquisition()
        
        self.height = height
        self.width = width

    def get_state(self):
        '''
            Threading : False
            Trimming : True
            return image dim:[1, height, width]
        '''
        _np_list = []
        self.cam.get_image(self.img)
        
        # self.data = np.array(Image.fromarray(np.uint8(self.img.get_image_data_numpy())).resize((self.width, self.height), resample=Image.BICUBIC))
        
        # Trimming
        image = self.img.get_image_data_numpy()
        image_trim = image[:, 64:576] # [512, 640] -> [512, 512]へ変換(array[:, 64:64+512])
        self.data = np.array(Image.fromarray(np.uint8(image_trim)).resize((self.width, self.height), resample=Image.BICUBIC))

        _np_list.append(self.data)

        next_state = np.array(_np_list)

        return next_state

    def end_hsc(self):
        self.cam.stop_acquisition()
        self.cam.close_device()