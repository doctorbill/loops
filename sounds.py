import math
import os
import shlex
import subprocess
import tempfile

import demucs.separate
import librosa
import matplotlib.pyplot as plt
import mutagen
import numpy as np
import soundfile as sf
import yt_dlp
from skimage.feature import match_template

def pretty_time_delta(seconds):
    seconds_fraction = seconds - int(seconds)
    seconds = int(seconds)
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    seconds += seconds_fraction
    if days > 0:
        return '%dd%dh%dm%.1fs' % (days, hours, minutes, seconds)
    elif hours > 0:
        return '%dh%dm%.1fs' % (hours, minutes, seconds)
    elif minutes > 0:
        return '%dm%.1fs' % (minutes, seconds)
    else:
        return '%.1fs' % (seconds,)

def rescale(x, low=0.0, high=1.0):
    y = np.copy(x)
    y -= np.amin(y)
    if y.max() != 0:
        y *= (high/y.max())
    return y

def change_speed_mono(data, start_rate=1.0, end_rate=1.0):
    output = []
    position = 0
    last_position = 0
    zeros = np.flatnonzero(librosa.zero_crossings(data))
    while position < len(data) - 1:
        last_position = position
        if len(zeros) < 1000:
            position = len(data) - 1
        else:
            position = zeros[1000]
            zeros = zeros[1001:]
        chunk = data[last_position:position]
        progress = (last_position + position) / (2 * len(data))
        output.append(librosa.resample(chunk, orig_sr=44100, target_sr=44100/(start_rate + (end_rate-start_rate)*progress), scale=True))
    return np.concatenate(output)

def frequency_map(y, start_semitones, end_semitones):
    steps = range(0, len(y), 4410)
    delta = (end_semitones - start_semitones) / len(steps)
    semitones = start_semitones
    frequency_map = []
    for step in steps:
        frequency_map.append((step, 2 ** (semitones / 12)))
        semitones += delta
    return frequency_map

def change_pitch(y, sr, start_semitones=0, end_semitones=0):
    frequency_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
    for x in frequency_map(y, start_semitones, end_semitones):
        frequency_file.write(f'{x[0]} {x[1]}\n')
    frequency_file.close()
    fd, infile = tempfile.mkstemp(suffix='.wav')
    os.close(fd)
    fd, outfile = tempfile.mkstemp(suffix='.wav')
    os.close(fd)
    sf.write(infile, y, sr)
    arguments = ['rubberband', '-q', '--fine', '--formant', '--freqmap', frequency_file.name, infile, outfile]
    subprocess.check_call(arguments, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    y_stretch, _ = sf.read(outfile, always_2d=True, dtype=y.dtype)
    if y.ndim == 1:
        y_stretch = np.squeeze(y_stretch)
    return y_stretch

class Loop:
    """Represents a loopable region of a Clip."""
    def __init__(self, start, end, beats, clip, score):
        self.start = start
        self.end = end
        self.beats = beats
        self.clip = clip
        self.score = score

    def fix_clip_bpm(self):
        # Adjust the clip's bpm so that self.beats is a whole number.
        # target_beats = round(self.beats)
        # target_seconds_per_beat = self.clip.duration / target_beats
        # target_bpm = 60 / target_seconds_per_beat
        # self.clip.bpm = target_bpm
        # self.beats = target_beats
        pass # TODO: Figure out if this is actually helpful.
    
    def clip_at_least(self, beats):
        # Return a clip of this loop that's at least the given number of beats long.
        copies = math.ceil(beats / self.beats)
        return AudioClip.append([self.clip] * copies)

class Timeline:
    '''Represents a sequence of clips, with offsets. An offset of None means we haven't decided where to put it yet (though the order is fixed).)'''
    def __init__(self, clips=None):
        clips = clips or []
        self.clips_and_offsets = [(c, None) for c in clips]

    def is_empty(self):
        return len(self.clips_and_offsets) == 0

    def end(self):
        return max([0.0] + [at_time + clip.duration for clip, at_time in self.clips_and_offsets if at_time is not None])

    def add(self, clip, at_time):
        self.clips_and_offsets.append((clip, at_time))

    def append(self, clip):
        self.add(clip, self.end())
    
    def export(self, path):
        AudioClip.mix(self.clips_and_offsets).save(path)

ydl_opts = {
    'outtmpl': './.isoflow_cache/%(id)s.%(ext)s',
    'format': 'm4a/bestaudio/best',
    'embed-metadata': True,
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'mp3'
    },
    {
        'key': 'FFmpegMetadata'
    }
    ]
}

class AudioClip:
    """Represents a clip of audio."""
    def __init__(self, source=None, data=None, name=None, bpm=None, is_slice=False, metadata=None, dirty=False):
        self.source = source
        self.name = name or (self.source or 'unknown').split('/')[-1].split('.')[0]
        if data is not None:
            self._data = data
        else:
            if self.source.startswith('https://'):
                cached = f'./.isoflow_cache/{self.source.split("=")[-1]}.mp3'
                if not os.path.exists(cached):
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        error_code = ydl.download([self.source])
                self.source = cached
            self._data = librosa.load(self.source, sr=44100, mono=False)[0]
            try:
                self.name = mutagen.File(self.source).tags['TIT2'].text[0].replace(' (Official Video)', '').replace(' (Official Audio)', '')
            except:
                pass
        # If self._data is mono, convert it to stereo.
        if self._data.ndim == 1:
            self._data = np.array([self._data, self._data])
        # bpm can be a number, or a pair of numnbers (start_bpm, end_bpm).
        self.bpm = bpm or np.mean(librosa.feature.tempo(y=self._data, sr=44100)) # Average of all channels.
        self.duration = self._data.shape[1]/44100
        self.cached_stfts = None # One stft per channel.
        self.cached_spectrogram = None
        self.is_slice = is_slice
        self.metadata = metadata or {}
        self.intro_loops = []
        self.outro_loops = []
        self.dirty = dirty or False

    def unload(self):
        if not self.dirty:
            self._data = None # Can be reloaded later
        return self

    def reload(self):
        if not self.dirty:
            self._data = librosa.load(self.source, sr=44100, mono=False)[0]
        return self

    def copy(self):
        return AudioClip(data=np.copy(self._data), name=self.name, bpm=self.bpm, is_slice=self.is_slice, metadata=self.metadata)

    def with_volume_fade(self, direction='in', duration=-1):
        clip = self.copy()
        if duration == -1:
            duration = clip.duration
        fade_length = round(duration*44100)
        if direction == 'in':
            fade = np.linspace(0, 1, fade_length)
            # Pad to the length of the clip.
            fade = np.pad(fade, (0, clip._data.shape[1]-fade_length), 'constant', constant_values=1)
        else:
            fade = np.linspace(1, 0, fade_length)
            # Pad to the length of the clip.
            fade = np.pad(fade, (clip._data.shape[1]-fade_length, 0), 'constant', constant_values=1)
        fade = np.array([fade, fade]) # Make it stereo.
        clip._data *= fade
        clip.dirty = True
        return clip

    @classmethod
    def mix(cls, clips_and_offsets):
        # Mix a list of clips together, padding the start of each clip with silence so that it starts at a given offset time (in seconds).
        padded_clips = [np.pad(c._data, ((0, 0), (round(offset*44100), 0)), 'constant') for c, offset in clips_and_offsets]
        # Also pad ends of clips, to make them all the same shape...
        length = max([c.shape[1] for c in padded_clips])
        padded_clips = [np.pad(c, ((0, 0), (0, length-c.shape[1])), 'constant') for c in padded_clips]
        mixed = np.sum(padded_clips, axis=0)
        volume = np.max(np.abs(mixed))
        mixed /= volume
        #unique_bpms = set([c.bpm for c, offset in clips_and_offsets]) # FIXME
        bpm = None #if len(unique_bpms) > 1 else unique_bpms.pop()
        return cls(data=mixed, bpm=bpm, dirty=True)

    @classmethod
    def append(cls, clips):
        # Append a list of clips together.
        data = np.concatenate([c._data for c in clips], axis=1)
        unique_bpms = set([c.bpm for c in clips])
        bpm = None if len(unique_bpms) > 1 else unique_bpms.pop()
        return cls(data=data, bpm=bpm, dirty=True)

    @classmethod
    def from_stfts(cls, stfts): # Might need length parameter too.
        channels_data = [librosa.istft(stft) for stft in stfts]
        return cls(data=np.array(channels_data), dirty=True)

    def seconds_per_beat(self):
        beats_per_second = self.bpm / 60
        return 1 / beats_per_second

    def bandpass_stft(self, low_hz, high_hz, fft_hop_length=512, n_fft=2048):
        # Return a new clip with only the frequencies between low_hz and high_hz, by zeroing out the rest of the STFT.
        stfts = self.stfts(fft_hop_length=fft_hop_length, n_fft=n_fft)
        frequencies = librosa.fft_frequencies(sr=44100, n_fft=n_fft)
        low_bin = np.argmin(np.abs(frequencies-low_hz))
        high_bin = np.argmin(np.abs(frequencies-high_hz))
        for stft in stfts:
            stft[:low_bin, :] = 0
            stft[high_bin:, :] = 0
        return AudioClip.from_stfts(stfts)

    def save(self, path, with_click_track=False):
        if with_click_track:
            tempo, click_times = librosa.beat.beat_track(y=librosa.to_mono(self._data), sr=44100, units='time', start_bpm=self.bpm)
            # print(f'Tempo = {tempo:.3f}s')
            AudioClip.mix([[self, 0], [AudioClip(data=librosa.clicks(times=click_times, sr=44100)), 0]]).save(path)
        else:
            return sf.write(path, np.transpose(self._data), 44100)

    def slice(self, start=0, end=-1, duration=-1):
        if end == -1 and duration != -1:
            end = start + duration
        start *= 44100
        if end != -1:
            end *= 44100
        clip = AudioClip(data=self._data[:, round(start):round(end)], bpm=self.bpm, is_slice=True, dirty=True)
        return clip

    def change_pitch(self, start_semitones=0, end_semitones=0): # Does not change speed.
        return AudioClip(data=np.transpose(change_pitch(np.transpose(self._data), 44100, start_semitones=start_semitones, end_semitones=end_semitones)),
                         bpm=self.bpm, metadata=self.metadata | {'start_semitones': start_semitones, 'end_semitones': end_semitones}, dirty=True)

    def change_speed(self, start_rate=1.0, end_rate=1.0): # Note: This also changes pitch.
        if start_rate == end_rate:
            bpm = self.bpm*start_rate
        else:
            bpm = [self.bpm*start_rate, self.bpm*end_rate]
        left = change_speed_mono(self._data[0], start_rate, end_rate)
        right = change_speed_mono(self._data[1], start_rate, end_rate)
        if len(left) < len(right):
            left = np.pad(left, (0, len(right) - len(left)))
        if len(right) < len(left):
            right = np.pad(right, (0, len(left) - len(right)))
        new_data = np.array([left, right])
        return AudioClip(data=new_data, bpm=bpm, metadata=self.metadata | {'start_rate': start_rate, 'end_rate':end_rate}, dirty=True)

    def stfts(self, fft_hop_length=512, n_fft=2048):
        if self.cached_stfts is None:
            self.cached_stfts = []
            for channel in range(self._data.shape[0]):
                self.cached_stfts.append(librosa.stft(y=self._data[channel, :], hop_length=fft_hop_length, n_fft=n_fft))
        return self.cached_stfts

    def spectrogram(self, fft_hop_length=512, n_fft=2048):
        if self.cached_spectrogram is None:
            D = np.abs(librosa.stft(y=librosa.to_mono(self._data), hop_length=fft_hop_length, n_fft=n_fft))**2
            self.cached_spectrogram = librosa.power_to_db(librosa.feature.melspectrogram(S=D, sr=44100, hop_length=fft_hop_length, n_fft=n_fft), ref=np.max)
        return self.cached_spectrogram

    def energy(self, smooth=True):
        if self.is_slice:
            print('WARNING: energy() called on a clip that is a slice. This is probably wrong.')
        sg = self.spectrogram()
        global_average_db = np.mean(sg) # TODO: Store this on the Source object, so we don't have to warn about slicing.
        # For each instant in time, find the average db value of all frequency buckets, and subtract the global average.
        energy = np.mean(sg, axis=0) - global_average_db
        if smooth:
            energy = np.convolve(energy, np.ones(2000)/2000, mode='same') # Moving average.
        energy /= np.max(np.abs(energy)) # Normalize.
        return energy

    def start_and_end_of_positive_energy(self):
        # Find the first offset where energy becomes positive, and the last offset before it becomes negative (in seconds).
        energy = self.energy()
        start = 0
        end = len(energy)-1
        while energy[start] < 0:
            start += 1
        while energy[end] < 0:
            end -= 1
        return librosa.frames_to_time(start, sr=44100, hop_length=512), librosa.frames_to_time(end, sr=44100, hop_length=512)

    def display(self, fft_hop_length=512, n_fft=2048):
        sg = self.spectrogram(fft_hop_length=fft_hop_length, n_fft=n_fft)
        energy = self.energy()
        fig, ax = plt.subplots(nrows=2, sharex=True)
        times = librosa.times_like(energy, sr=44100)
        ax[0].plot(times, energy, label='Energy')
        ax[0].set(xticks=[])
        ax[0].legend()
        ax[0].label_outer()
        ax[0].axhline(y=0, color='r')
        img = librosa.display.specshow(sg, y_axis='mel', x_axis='time', ax=ax[1], sr=44100)
        ax[1].set_title('Power spectrogram')
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        plt.show()

    # NOTE: This is really slow. In reality we'll only want to run it on the last 30-seconds of track1, and the first 30-seconds of track2.
    def separated_sources(self):
        demucs.separate.main(opts=shlex.split('7CTJcHjkq0E.wav')) # TODO: Write to a temporary file, and return the results.
        return self

    def best_offset_of(self, other, fft_hop_length=512, n_fft=2048, adjust_bpm=False, adjust_pitch=False):
        # Find the best offset of other in self, using normalized cross-correlation.
        # If adjust_bpm is True, adjust the bpm of other to match self.
        # If adjust_pitch is True, create 12 copies of other, each shifted by a semitone, and find the best of all of them.
        # print(f'My bpm is {self.bpm:.2f}, other is {other.bpm:.2f}')
        if adjust_bpm:
            other = other.change_speed(self.bpm/other.bpm, self.bpm/other.bpm)
        if adjust_pitch:
            others = [other.change_pitch(semitones, semitones) for semitones in range(-6, 7)]
        else:
            others = [other]
        results = []
        for other in others:
            matches = match_template(self.spectrogram(fft_hop_length=fft_hop_length, n_fft=n_fft), other.spectrogram(fft_hop_length=fft_hop_length, n_fft=n_fft))[0]
            best = np.argmax(matches)
            best_time = librosa.frames_to_time(best, sr=44100, hop_length=fft_hop_length)
            # print(f'Best match for {other.metadata} is at {best_time:.3f}s, with score {matches[best]:.3f}')
            results += [(best_time, matches[best], other)]
        results.sort(key=lambda x: x[1], reverse=True)
        return results[0]

    # TODO: Try looping different frequencies separately, and then mixing them together.
    # Experiments TODO:
    # Can we shift parts of the STFT around to change the timing of certain frequencies?

    def find_loopable_region(self, max_duration=30, after=0, before=None, skip=0.5):
        # Find the best of all the regions of self that loop well.
        loops = self.find_loopable_regions(max_duration=max_duration, after=after, before=before, skip=skip)
        #print(f'New best loop: score={best_score:.3f}, {pretty_time_delta(best_loop_start)} -> {pretty_time_delta(best_loop_found_at)} ({pretty_time_delta(best_loop_found_at-best_loop_start)}, {beats:.1f} beats long)')
        return sorted(loops, key=lambda x: x.score, reverse=True)[0]

    def find_loopable_regions_by_beat_count(self, max_duration=30, after=0, before=None, skip=0.5):
        loops = self.find_loopable_regions(max_duration=max_duration, after=after, before=before, skip=skip)
        best_by_beat_count = {}
        for loop in loops:
            beats = loop.beats
            if beats not in best_by_beat_count or loop.score > best_by_beat_count[beats].score:
                best_by_beat_count[beats] = loop
        return sorted(best_by_beat_count.values(), key=lambda x: x.score, reverse=True)

    def find_loopable_regions(self, max_duration=30, after=0, before=None, skip=0.5):
        # Find regions of self that loop well.
        loops = []
        compare_duration = 2
        before = before or self.duration
        earliest_start = int(after+0.5)
        latest_start = min(int(self.duration - max_duration - compare_duration), int(before))
        for loop_start in np.arange(earliest_start, latest_start, skip):
            if self.energy()[librosa.time_to_frames(loop_start, sr=44100)] > 0:
                loop_clip = self.slice(start=loop_start, duration=compare_duration)
                offset, score, _ = self.slice(start=loop_start+compare_duration, duration=max_duration+compare_duration).best_offset_of(loop_clip)
                loop_found_at = loop_start+compare_duration+offset
                beats = (loop_found_at - loop_start)/self.seconds_per_beat()
                loops.append(Loop(start=loop_start, end=loop_found_at, score=score, beats=beats, clip=self.slice(start=loop_start, end=loop_found_at)))
        return loops
