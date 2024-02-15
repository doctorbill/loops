#!/usr/bin/env python3

# Find loopable regions of audio

import sys
import sounds
import time
import argparse

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
    print(f'Positive energy region: {sounds.pretty_time_delta(positive_energy_starts)} -> {sounds.pretty_time_delta(positive_energy_ends)}')
    print('Finding best loopable regions...')
    loops = s.find_loopable_regions_by_beat_count(after=positive_energy_starts,
                                                  before=positive_energy_ends,
                                                  max_duration=max_duration,
                                                  skip=skip)
    for n, loop in enumerate(loops[:5]): # Best 5 loops
        print(f'Loop {n+1}: score={loop.score:.3f}, beats={loop.beats:.1f}, bpm={loop.clip.bpm:.2f}, duration={loop.clip.duration:.2f}s')
        repeats = 3
        sounds.AudioClip.append([loop.clip] * repeats).save(f'{s.name}-{n+1}-{round(loop.beats)}beats.wav', with_click_track=args.click)
    ended = time.time()
    print(f'Finished in {sounds.pretty_time_delta(ended-started)}.')
    s = s.unload()
