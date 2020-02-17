# SMH: Smart Monitor for Homes

## Inspiration
At the opening ceremony Daniel brought up the idea of making a fake news app but Ben misheard because it was too loud and thought he said 'baby sounds'. So we started making an app that tried to interpret baby sounds and predict what they meant (if the baby was hungry, tired, etc.)

However, the data available was quite lackluster, so instead we repurposed our classifier to instead detect sounds associated with home emergencies (broken glass, smoke alarm, etc) and alert users accordingly.

## What it does
A smart monitor (Google AIY kit with Raspberry Pi) listens for sounds associated with emergencies. If such a sound is detected, a notification will be sent to the mobile app notifying users of a possible emergency.

## How we built it
First, we used the AIY voice kit to capture audio and save it locally in the monitor.

The classifier runs internally in the monitor. Using pandas, librosa, and numpy, it vectorizes the captured audio files and trains a neural network to predict a) if a sound is an emergency, and b) what type of emergency it is.

If an emergency gets detected, the information (time, type, location, etc) gets uploaded to the server (a Flask app running in a basement at Berkeley) which then sends the data to the app.

The app is a React Native client which sends a push notification and displays the incident on a list.

## Challenges we ran into
Getting good sound data was pretty hard. Most of it is either locked under a commercial paywall or poor quality.

## Accomplishments that we're proud of
We had fun and the food was pretty good :)

## What we learned
How to make really cool backends, the power of SSH, React, how to process sound for machine learning, how to connect some seemingly unconnected technologies (Python, JS, Bash, hardware)

## What's next for SMH: Smart Monitor for Homes
Probably nothing honestly. We were just here to chill and have fun so this is kind of just a proof of concept.
