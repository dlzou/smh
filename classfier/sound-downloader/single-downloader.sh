SAMPLE_RATE=22050
outname=$2
youtube-dl $1 \
    --quiet --extract-audio --audio-format wav \
    --output "./testing/$outname.%(ext)s"
yes | ffmpeg -loglevel quiet -i "./testing/$outname.wav" -ar $SAMPLE_RATE \
      -ss "$3" -to "$4" "./testing/${outname}_out.wav"
    mv "./testing/${outname}_out.wav" "./testing/$outname.wav"