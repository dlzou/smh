import requests
import threading
import traceback
import time
from aiy.board import Board, Led
from aiy.voice.audio import AudioFormat, record_file, play_wav

TEST_SOUND = '/usr/share/sounds/alsa/Front_Center.wav'
FILENAME = 'recording.wav'

# IGNORE, NOT USED
class UniqueID:
    def __init__(self):
        self.active_ids = []

    def get_new_id(self):
        if len(self.active_ids) == 0:
            self.active_ids.append(0)
            return 0
        for index, id in enumerate(self.active_ids):
            if index != id:
                self.active_ids = self.active_ids[:index] + [index] + self.active_ids[index:]
                return index
        self.active_ids.append(index + 1)
        return index + 1

    def release_id(self, id):
        self.active_ids.remove(id)


def main():
    print('Playing test sound...')
    play_wav(TEST_SOUND)

    with Board() as board:
        while True:
            board.led.state = Led.OFF
            print('Press button to start recording...')
            board.button.wait_for_press()
            board.led.state = Led.ON

            done = threading.Event()
            board.button.when_pressed = done.set

            record_file(AudioFormat.CD, filename=FILENAME, wait=wait(done), filetype='wav')

            # run classifier
            # state = 'hungry'
            # print(state)
            # payload = {'type': state}
            # headers = {'Accept': 'application/json', 'Content-Type': 'application/json'}
            # r = requests.post('http://bigrip.ocf.berkeley.edu:5000/notify', json=payload, headers=headers)
            with open(FILENAME, 'rb') as file:
                r = requests.post('http://bigrip.ocf.berkeley.edu:5000/sendaudio', data=file)
            print(r.status_code)

def wait(done):
    def _helper():
        start = time.monotonic()
        while not done.is_set():
            duration = time.monotonic() - start
            print('Recording: %.02f seconds [Press button to stop]' % duration)
            time.sleep(0.5)
    return _helper


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
