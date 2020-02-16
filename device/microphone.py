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

        print('Press button to start recording...')
        board.button.wait_for_press()

        done = threading.Event()
        board.button.when_pressed = done.set

        def wait():
            start = time.monotonic()
            while not done.is_set():
                duration = time.monotonic() - start
                print('Recording: %.02f seconds [Press button to stop]' % duration)
                time.sleep(0.5)

        record_file(AudioFormat.CD, filename=FILENAME, wait=wait, filetype='wav')
        # board.button.wait_for_press()

        # run classifier
        baby_state = 'hungry'
        print(baby_state)
        payload = {'type': baby_state}
        headers = {'content-type': 'application/json'}
        r = requests.post('http://bigrip.ocf.berkeley.edu:5000/notify', data=payload, headers=headers)
        print(r.status_code)

if __name__ == '__main__':
    try:
        main()
    except:
        print("Whoops")