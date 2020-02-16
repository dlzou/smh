import requests
import threading
import traceback
import time
from aiy.board import Board, Led
from aiy.voice.audio import AudioFormat, record_file, play_wav

TEST_SOUND = '/usr/share/sounds/alsa/Front_Center.wav'
FILENAME = 'recording.wav'

def main():
    print('Playing test sound...')
    play_wav(TEST_SOUND)

    with Board() as board:
        while True:
            board.led.state = Led.BEACON_DARK
            print('Press button to start recording...')
            board.button.wait_for_press()
            board.led.state = Led.ON

            done = threading.Event()
            board.button.when_pressed = done.set

            def wait():
                start = time.monotonic()
                while not done.is_set():
                    duration = time.monotonic() - start
                    print('Recording: %.02f seconds [Press button to stop]' % duration)
                    time.sleep(0.5)

            record_file(AudioFormat.CD, filename=FILENAME, wait=wait, filetype='wav')

            # run classifier
            state = 'hungry'
            print(state)
            payload = {'type': state}
            headers = {'Accept': 'application/json', 'Content-Type': 'application/json'}
            r = requests.post('http://bigrip.ocf.berkeley.edu:5000/notify', json=payload, headers=headers)
            print(r.status_code)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
