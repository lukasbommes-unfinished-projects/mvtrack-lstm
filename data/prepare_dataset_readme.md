1) Download and extract 2D MOT 2015, MOT 2016 and MOT 2017 datasets from https://motchallenge.net/ in this directory
2) Set the options "CODEC" and "DATASET" options in `video_write.py` and `extract_mvs.py`. CODEC can be either "h264" or "mpeg4" and determines which the encoder used to encode the individual frames into videos. The DATASET option specifies which of the three datasets (MOT15, MOT16 or MO17) is processed when calling the scripts.
3) Additional options can be tuned in `seqinfo.py`
4) Inside the docker container run `python video_write.py` to encode frames into videos at specified scales and with specified encoder.
5) Inside the docker container run `python extract_mvs.py` to extract motion vectors from the previously generated videos.
