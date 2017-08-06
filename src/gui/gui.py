from PyQt5 import uic, QtWidgets
from PyQt5.QtCore import QThread
from PyQt5.QtGui import QIcon, QPixmap
import pyaudio
import wave

from datetime import datetime

if __name__ == '__main__':
    import sys
    sys.path.append("..")

from nn.neural_network import train, real_time_predict, predict, predict_from_file
from nn.recorder import record_to_file, AudioFile
from feature.sigproc import task
from nn.recorder import record_to_file

def print_label(text, character="*"):
    star = int((80-len(text))/2)
    return (character*star + text + character*star)

class PlayThread(QThread):
    def __init__(self, path):
        QThread.__init__(self)
        self.path = path

    def run(self):
        wavFile = wave.open("test.wav", 'rb')
        audio = pyaudio.PyAudio()

        # start Recording
        stream = audio.open(format = pyaudio.paInt16, channels = 1, rate = 8000,
                            frames_per_buffer = 1024, output=True)
        frames = []
        data = wavFile.readframes(1024)
        while data:
            stream.write(data)
            data = wavFile.readframes(1024)

        # stop Recording
        stream.close()
        audio.terminate()

class RecordThread(QThread):
    def __init__(self, main):
        QThread.__init__(self)
        self.main = main

    def run(self):
        audio = pyaudio.PyAudio()

        # start Recording
        stream = audio.open(format = pyaudio.paInt16, channels = 1, rate = 8000,
                            frames_per_buffer = 1024, input=True)
        frames = []

        while self.main.stop == False:
            data = stream.read(1024)
            frames.append(data)

        # stop Recording
        stream.stop_stream()
        stream.close()
        audio.terminate()

        waveFile = wave.open("test.wav", 'wb')
        waveFile.setnchannels(1)
        waveFile.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        waveFile.setframerate(8000)
        waveFile.writeframes(b''.join(frames))
        waveFile.close()

class TrainThread(QThread):
    def __init__(self, main):
        QThread.__init__(self)
        self.main = main

    def run(self):
        self.main.start_train_thread()

class PredictionThread(QThread):
    def __init__(self, main):
        QThread.__init__(self)
        self.main = main

    def run(self):
        self.main.start_predict_thread()

class TestPredictionThread(QThread):
    def __init__(self, main):
        QThread.__init__(self)
        self.main = main

    def run(self):
        self.main.start_test_predict_thread()

class Ui(QtWidgets.QDialog):
    def __init__(self, uipath="user_interface.ui", verbose=True):
        super(Ui, self).__init__()
        uic.loadUi(uipath, self)
        self.output = ""
        self.verbose = verbose
        self.stop = True

        #Thread
        self.record_th = RecordThread(self)
        self.record_th.finished.connect(self.record_fin)
        self.play_th = PlayThread(self)
        self.play_th.finished.connect(self.play_fin)

        self.train_th = TrainThread(self)
        self.train_th.finished.connect(self.train_fin)

        self.predict_th = PredictionThread(self)
        self.predict_th.finished.connect(self.predict_fin)

        self.test_predict_th = TestPredictionThread(self)
        self.test_predict_th.finished.connect(self.test_predict_fin)

        # UI Initializes
        self.btn_train.clicked.connect(self.start_train)
        self.btn_predict.clicked.connect(self.start_predict)
        self.btn_test_predict.clicked.connect(self.start_test_predict)
        self.btn_enroll.clicked.connect(self.start_enroll)
        self.btn_record.clicked.connect(self.start_record)
        self.btn_play.clicked.connect(self.play_wav)
        self.show_pic("Anynomous")
        # Show the form
        self.show()

    def verbose_changed(self):
        self.verbose = self.checkBox_verbose.isChecked()

    def start_conv_mode(self):
        # self.stop = False
        # self.lbl_username.setText("Surendra")
        # self.show_pic("Anynomous")
        # self.conv_mode = True
        # self.btn_start_conv_mode.setEnabled(False)
        # self.btn_stop_conv_mode.setEnabled(True)
        # record_to_file("test.wav", RECORD_SECONDS=1)
        from feature.sigproc import record_and_save
        record_and_save("test.wav", "files/test/test.wav")

    def stop_conv_mode(self):
        self.stop = True
        self.record_th.wait()
        self.lbl_username.setText("Stopped")
        self.btn_start_conv_mode.setEnabled(True)
        self.btn_stop_conv_mode.setEnabled(False)
        self.show_pic("Anynomous")
        self.conv_mode = False

    def start_record(self):
        if self.stop:
            # Now start Recording
            self.lbl_status.setText("Recording file")
            self.btn_record.setText("Stop")
            self.stop = False
            self.record_th.start()
        else:
            self.lbl_status.setText("Recording stopped.")
            self.stop = True
            self.record_th.wait()
            self.btn_record.setText("Record")

    def record_fin(self):
        # Record fin perform other task
        #perform calculations
        if self.conv_mode:
            self.show_pic(predict_from_file())
            self.record_th.start()

    def play_wav(self):
        self.btn_play.setEnabled(False)
        self.lbl_status.setText("Playing file...")
        self.play_th.start()

    def play_fin(self):
        self.btn_play.setEnabled(True)

    def start_enroll(self):
        filename = self.line_edit_filename.text()
        if len(filename) < 5:
            print("Filename too short")
            self.lbl_status.setText("Filename too short")
            return
        task("test.wav", "files/wav/"+filename+".wav")
        self.lbl_status.setText("File enrolled.")

    def start_train(self):
        self.output = print_label("Training started")
        self.train_th.start()
        self.btn_train.setEnabled(False)

        self.lbl_output_train.setText(self.output)
        self.start_time = datetime.now()

    def start_train_thread(self):
        self.output += "\n"

        self.output += train("files/wav", "model.pkl", layer_sizes=(5000,), verbose=self.verbose)

    def train_fin(self):
        end_time = datetime.now()
        elapsed_time = end_time-self.start_time
        self.output +="\nElapsed time = %s\n"%elapsed_time
        self.output += print_label("Training finished")

        self.btn_train.setEnabled(True)

        self.lbl_output_train.setText(self.output)

    def start_predict(self):
        self.btn_predict.setEnabled(False)
        self.lbl_username.setText('Recording...')
        pixmap = QPixmap('files/avatar/recording.png')
        self.lbl_userphoto.setPixmap(pixmap)
        self.predict_th.start()

    def start_predict_thread(self):
        self.output = real_time_predict(speech_len=8, verbose=self.verbose)

    def predict_fin(self):
        self.show_pic(self.output)
        self.btn_predict.setEnabled(True)

    def start_test_predict(self):
        self.output = print_label("Predicting")
        self.btn_test_predict.setEnabled(False)
        self.test_predict_th.start()
        self.lbl_output_file_predict.setText(self.output)

    def start_test_predict_thread(self):
        self.output += "\n"
        self.output += predict("model.pkl", "files/test/wav", threshold=0.95, verbose=self.verbose)

    def test_predict_fin(self):
        self.btn_test_predict.setEnabled(True)
        self.output += "\n"
        self.output += print_label("Predict finished.")
        self.lbl_output_file_predict.setText(self.output)

    def show_pic(self, avatar_name):
        # Display picture of the known user
        self.lbl_username.setText(avatar_name)
        pixmap = QPixmap('files/avatar/'+avatar_name+'.png')
        self.lbl_userphoto.setPixmap(pixmap)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    import os
    os.chdir("..")
    window = Ui(uipath="gui/user_interface.ui")
    sys.exit(app.exec_())
