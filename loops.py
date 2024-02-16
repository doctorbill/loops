#!/usr/bin/env python3

# To set everything up, do pip3 install -r ./requirements.txt

# Find loopable regions of audio
# Good examples:
# ./loops.py -i 'https://www.youtube.com/watch?v=fHI8X4OXluQ'
#     Pretty basic, but does a great job!
# ./loops.py -i 'https://www.youtube.com/watch?v=T0_zzCLLRvE'
#     I especially like this one because it finds a completely seemless loop with an odd number of beats!
# ./loops.py -i 'https://www.youtube.com/watch?v=fJ9rUzIMcZQ'
#     Pretty great/funny!
# ./loops.py -i 'https://www.youtube.com/watch?v=N_3_skKeGCc'
#     I think it does a nice job, given how little rhythmic information there is in the source.

import argparse
import sys
import time

import sounds

parser = argparse.ArgumentParser(description="Find loopable regions in audio")
parser.add_argument("--click", help="Add a click track to outputs", action="store_true")
parser.add_argument("--slow", help="Spend more time processing (for hopefully better results)", action="store_true")
parser.add_argument("--slowest", help="Spend much more time processing (for hopefully better results)", action="store_true")
parser.add_argument("--fast", help="Spend less time processing (for probably worse results)", action="store_true")
parser.add_argument("--fastest", help="Spend much less time processing (for probably much worse results)", action="store_true")
parser.add_argument("--display", help="Display some helpful debugging information for each source", action="store_true")
parser.add_argument("-i", "--input", help="Input files or urls", required=True, nargs="+")
args = parser.parse_args()

skip = 0.5
if args.slow:
    skip = 0.25
if args.slowest:
    skip = 0.1
if args.fast:
    skip = 1
if args.fastest:
    skip = 5

if args.display:
    for f in args.input:
        s = sounds.AudioClip(f)
        print(f'  {s.name}: {s.bpm:.2f} bpm')
        s.display()
    sys.exit()

sources = [sounds.AudioClip(f).unload() for f in args.input]

print('Inputs:')
for s in sources:
    print(f'  {s.bpm:.2f} bpm:  {s.name}')

for s in sources:
    s = s.reload()
    print(f'Finding loops for {s.name}...')
    max_duration = 20 # Longest loop duration (seconds)
    started = time.time()
    positive_energy_starts, positive_energy_ends = s.start_and_end_of_positive_energy()
    print(f'Positive energy: {sounds.pretty_time_delta(positive_energy_starts)} -> {sounds.pretty_time_delta(positive_energy_ends)}')
    print('Finding best loopable regions...')
    loops = s.find_loopable_regions_by_beat_count(after=positive_energy_starts,
                                                  before=positive_energy_ends,
                                                  max_duration=max_duration,
                                                  skip=skip)
    for n, loop in enumerate(loops[:5]): # Best 5 loops
        filename = f'{s.name}-{n+1}-{round(loop.beats)}beats.wav'
        print(f'{n+1}: score={loop.score:.3f}, beats={loop.beats:.1f}, duration={loop.clip.duration:.2f}s -> {filename}')
        repeats = 3 # Change this to make each output loop longer or shorter
        sounds.AudioClip.append([loop.clip] * repeats).save(filename, with_click_track=args.click)
    ended = time.time()
    print(f'Finished in {sounds.pretty_time_delta(ended-started)}.')
    s = s.unload()
