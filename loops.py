#!/usr/bin/env python3

# Make aesthetically pleasing transitions between songs

# 1. Use an audio-matcher to figure out the best way to overlap the two tracks (possibly only using the drum track? Or drum+bass?)
# 2.1. The above step gets done in a loop that pitch-shifts the two tracks, and tries to find the best pitch-shift possible (try all semitones).
# 2.2. Another loop time-stretches the two tracks, and tries to find the best time-stretch possible.
# 3. Once a really good overlap is found, apply a transition strategy to the overlapping region, and crossfade the two tracks.

import sys
import sounds
import time
import argparse

parser = argparse.ArgumentParser(description="Make nice transitions between two or more songs")
parser.add_argument("--for-daw", help="Write output suitable for importing into a DAW", action="store_true")
parser.add_argument("--sharable", help="Write a sharable (short) output, containing just the transition", action="store_true")
parser.add_argument("--video", help="Write video output", action="store_true")
parser.add_argument("--no-sort-by-bpm", help="Switch off sorting sources by bpm (rising and then falling)", action="store_true")
parser.add_argument("--test-looping", help="Test looping (intro and outro) for each source file", action="store_true")
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

if not args.no_sort_by_bpm:
    sorted_sources = sorted(sources, key=lambda s: s.bpm)
    sources = []
    parity = 1
    # Arrange songs by bpm, ramping up and down
    while len(sorted_sources) > 0:
        if parity == 1:
            sources += [sorted_sources.pop()]
        else:
            sources = [sorted_sources.pop()] + sources
        parity *= -1

print('Inputs:')
for s in sources:
    print(f'{s.bpm:.2f} bpm:  {s.name}')

# Find intro and outro loops for each source...
for s in sources:
    s = s.reload()
    print(f'Finding loops for {s.name}...')
    max_duration = 20 # TODO: It probably makes sense to make this depend on the bpm of the song.
    started = time.time()
    positive_energy_starts, positive_energy_ends = s.start_and_end_of_positive_energy()
    print(f'Positive energy region: {sounds.pretty_time_delta(positive_energy_starts)} -> {sounds.pretty_time_delta(positive_energy_ends)}')
    print('Finding intro loopable region...')
    s.intro_loops = s.find_loopable_regions_by_beat_count(after=positive_energy_starts,
                                                         before=60+positive_energy_starts,
                                                         max_duration=max_duration,
                                                         skip=skip)[:2] # Best 2 intro loops
    for n, intro_loop in enumerate(s.intro_loops):
        intro_loop.fix_clip_bpm()
        print(f' Intro loop: score={intro_loop.score:.3f}, beats={intro_loop.beats}, bpm={intro_loop.clip.bpm:.2f}, duration={intro_loop.clip.duration:.2f}s')
        sounds.AudioClip.append([intro_loop.clip] * 3).save(f'{s.name}-intro-{n+1}-{round(intro_loop.beats)}beats.wav', with_click_track=True)
    print('Finding outro loopable region...')
    s.outro_loops = s.find_loopable_regions_by_beat_count(after=max(positive_energy_ends-60-max_duration, max([i.end for i in s.intro_loops])),
                                                         before=positive_energy_ends-max_duration,
                                                         max_duration=max_duration,
                                                         skip=skip)[:2] # Best 2 outro loops
    for n, outro_loop in enumerate(s.outro_loops):
        outro_loop.fix_clip_bpm()
        print(f' Outro loop: score={outro_loop.score:.3f}, beats={outro_loop.beats}, bpm={outro_loop.clip.bpm:.2f}, duration={outro_loop.clip.duration:.2f}s')
        sounds.AudioClip.append([outro_loop.clip] * 3).save(f'{s.name}-outro-{n+1}-{round(outro_loop.beats)}beats.wav', with_click_track=True)
    ended = time.time()
    print(f'Finished in {sounds.pretty_time_delta(ended-started)}.')
    s = s.unload()

timeline = sounds.Timeline()
previous_source = None
previous_intro = None
for source in sources:
    source = source.reload()
    print(f'Mixing {source.name}...')
    if previous_source:
        print(f'Mixing {previous_source.name} into {source.name}...')
        best = None
        for outro in previous_source.outro_loops:
            for intro in source.intro_loops:
                print(f'  Previous track outro: {outro.clip.bpm:.2f} bpm for {outro.clip.duration:.1f}s ({outro.beats} beats), Next track intro: {intro.clip.bpm:.2f} bpm for {intro.clip.duration:.1f}s ({intro.beats} beats). Need to adjust by {outro.clip.bpm/intro.clip.bpm:.2f}x')
                outro_looped_clip = outro.clip_at_least(2 * intro.beats)
                offset, score, clip = outro_looped_clip.best_offset_of(intro.clip, adjust_bpm=True, adjust_pitch=True)
                if best is None or score > best['score']:
                    best = {'score': score, 'offset': offset, 'clip': clip, 'intro': intro, 'outro': outro}
        print(f'Best transition: Score={best["score"]:.3f}, offset={best["offset"]:.3f}s, metadata={best["clip"].metadata}, intro score={best["intro"].score:.3f}, outro score={best["outro"].score:.3f}')
        # Things we need to mix:
        # 1. The body of the previous source, up to the end of the chosen outro loop
        # FIXME: The body's speed needs to be adjusted smoothly, to match the chosen outro loop's speed
        # if timeline.is_empty():
        #     timeline.append(previous_source.slice(0, best['outro'].start))
        # else:
        #     timeline.append(previous_source.slice(previous_intro.end, best['outro'].start))
        # 2. The chosen outro loop of the previous source, looped, with a volume fade-out
        timeline.append(outro_looped_clip) # We'll add it twice -- once here, by itself...
        end = timeline.end()
        timeline.append(outro_looped_clip.with_volume_fade('out')) # This time with the next intro, and a crossfade
        # 3. The chosen intro loop of the current source, looped, with a volume fade-in, with the correct offset
        timeline.add(sounds.AudioClip.append([best['intro'].clip]*3).with_volume_fade('in', best['intro'].clip.duration), end + best['offset'])
        previous_intro = best['intro']
    previous_source = source

# TODO: After all of the songs have been mixed, choose an outro loop for the last source, loop it, and fade it out.
timeline.export('mix.wav')
