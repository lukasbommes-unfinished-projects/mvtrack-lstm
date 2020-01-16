# This script converts each sequence's imaged of the MOT dataset into H.264 encoded video sequences
import os
import subprocess
from seqinfo import scales, frame_rates, dir_names


CODEC = "mpeg4"  # "h264" or "mpeg4"
DATASET = "MOT15"

if __name__ == "__main__":

    cwd = os.getcwd()
    for mode in ["train", "test"]:
        for dir_name, frame_rate in zip(dir_names[DATASET][mode], frame_rates[DATASET][mode]):
            os.chdir(os.path.join(cwd, DATASET, mode, dir_name, 'img1'))
            for scale in scales:
                if CODEC == "h264":
                    subprocess.call(['ffmpeg', '-y', '-r', str(frame_rate),
                        '-i', '%06d.jpg', '-c:v', 'libx264', '-vf',
                        'scale=iw*{}:ih*{}, pad=ceil(iw/2)*2:ceil(ih/2)*2'.format(scale, scale),
                        '-f', 'rawvideo', '../{}-{}-{}.mp4'.format(dir_name, CODEC, scale)])
                elif CODEC == "mpeg4":
                    subprocess.call(['ffmpeg', '-y', '-r', str(frame_rate),
                        '-i', '%06d.jpg', '-c:v', 'mpeg4', '-vf',
                        'scale=iw*{}:ih*{}, pad=ceil(iw/2)*2:ceil(ih/2)*2'.format(scale, scale),
                        '-f', 'rawvideo', '../{}-{}-{}.mp4'.format(dir_name, CODEC, scale)])
