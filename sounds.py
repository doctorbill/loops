import os

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

class Loop:
    """Represents a loopable region of a Clip."""
    def __init__(self, start, end, beats, clip, score):
        self.start = start
        self.end = end
        self.beats = beats
        self.clip = clip
        self.score = score

ydl_opts = {
    'outtmpl': './.ydl_cache/%(id)s.%(ext)s',
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
    def __init__(self, source=None, data=None, name=None, bpm=None, is_slice=False, dirty=False, samplerate=None):
        self.samplerate = samplerate or 44100
        self.source = source
        self.name = name or (self.source or 'unknown').split('/')[-1].split('.')[0]
        if data is not None:
            self._data = data
        else:
            if self.source.startswith('https://'):
                cached = f'./.ydl_cache/{self.source.split("=")[-1]}.mp3'
                if not os.path.exists(cached):
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        error_code = ydl.download([self.source])
                self.source = cached
            self._data = librosa.load(self.source, sr=self.samplerate, mono=False)[0]
            try:
                self.name = mutagen.File(self.source).tags['TIT2'].text[0].replace(' (Official Video)', '').replace(' (Official Audio)', '')
            except:
                pass
        # If self._data is mono, convert it to stereo.
        if self._data.ndim == 1:
            self._data = np.array([self._data, self._data])
        # bpm can be a number, or a pair of numnbers (start_bpm, end_bpm).
        self.bpm = bpm or np.mean(librosa.feature.tempo(y=self._data, sr=44100)) # Average of all channels.
        self.duration = self._data.shape[1]/self.samplerate
        self.cached_spectrogram = None
        self.is_slice = is_slice
        self.dirty = dirty or False

    def unload(self):
        if not self.dirty:
            self._data = None # Can be reloaded later
        return self

    def reload(self):
        if not self.dirty:
            self._data = librosa.load(self.source, sr=self.samplerate, mono=False)[0]
        return self

    def copy(self):
        return AudioClip(data=np.copy(self._data), name=self.name, bpm=self.bpm, is_slice=self.is_slice)

    @classmethod
    def mix(cls, clips_and_offsets):
        # Ensure clips are all the same sample rate
        if not 1 == len(set([c.samplerate for c, offset in clips_and_offsets])):
            raise Exception("cannot mix AudioClips with different sample rates")
        # Mix a list of clips together, padding the start of each clip with silence so that it starts at a given offset time (in seconds).
        padded_clips = [np.pad(c._data, ((0, 0), (round(offset*c.samplerate), 0)), 'constant') for c, offset in clips_and_offsets]
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

    def seconds_per_beat(self):
        beats_per_second = self.bpm / 60
        return 1 / beats_per_second

    def save(self, path, with_click_track=False):
        if with_click_track:
            tempo, click_times = librosa.beat.beat_track(y=librosa.to_mono(self._data), sr=self.samplerate, units='time', start_bpm=self.bpm)
            # print(f'Tempo = {tempo:.3f}s')
            AudioClip.mix([[self, 0], [AudioClip(data=librosa.clicks(times=click_times, sr=self.samplerate)), 0]]).save(path)
        else:
            return sf.write(path, np.transpose(self._data), self.samplerate)

    def slice(self, start=0, end=-1, duration=-1):
        if end == -1 and duration != -1:
            end = start + duration
        start *= self.samplerate
        if end != -1:
            end *= self.samplerate
        clip = AudioClip(data=self._data[:, round(start):round(end)], bpm=self.bpm, is_slice=True, dirty=True)
        return clip

    def spectrogram(self, fft_hop_length=512, n_fft=2048):
        if self.cached_spectrogram is None:
            D = np.abs(librosa.stft(y=librosa.to_mono(self._data), hop_length=fft_hop_length, n_fft=n_fft))**2
            self.cached_spectrogram = librosa.power_to_db(librosa.feature.melspectrogram(S=D, sr=self.samplerate, hop_length=fft_hop_length, n_fft=n_fft), ref=np.max)
        return self.cached_spectrogram

    def energy(self, smooth=True):
        if self.is_slice:
            print('WARNING: energy() called on a clip that is a slice. This is probably wrong.')
        sg = self.spectrogram()
        global_average_db = np.mean(sg)
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
        return librosa.frames_to_time(start, sr=self.samplerate, hop_length=512), librosa.frames_to_time(end, sr=self.samplerate, hop_length=512)

    def display(self, fft_hop_length=512, n_fft=2048):
        sg = self.spectrogram(fft_hop_length=fft_hop_length, n_fft=n_fft)
        energy = self.energy()
        fig, ax = plt.subplots(nrows=2, sharex=True)
        times = librosa.times_like(energy, sr=self.samplerate)
        ax[0].plot(times, energy, label='Energy')
        ax[0].set(xticks=[])
        ax[0].legend()
        ax[0].label_outer()
        ax[0].axhline(y=0, color='r')
        img = librosa.display.specshow(sg, y_axis='mel', x_axis='time', ax=ax[1], sr=self.samplerate)
        ax[1].set_title('Power spectrogram')
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        plt.show()

    def best_offset_of(self, other, fft_hop_length=512, n_fft=2048):
        # Find the best offset of other in self, using normalized cross-correlation.
        matches = match_template(self.spectrogram(fft_hop_length=fft_hop_length, n_fft=n_fft), other.spectrogram(fft_hop_length=fft_hop_length, n_fft=n_fft))[0]
        best = np.argmax(matches)
        best_time = librosa.frames_to_time(best, sr=self.samplerate, hop_length=fft_hop_length)
        return (best_time, matches[best], other)

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
        _energy = self.energy()
        for loop_start in np.arange(earliest_start, latest_start, skip):
            if _energy[librosa.time_to_frames(loop_start, sr=self.samplerate)] > 0:
                loop_clip = self.slice(start=loop_start, duration=compare_duration)
                offset, score, _ = self.slice(start=loop_start+compare_duration, duration=max_duration+compare_duration).best_offset_of(loop_clip)
                loop_found_at = loop_start+compare_duration+offset
                beats = (loop_found_at - loop_start)/self.seconds_per_beat()
                loops.append(Loop(start=loop_start, end=loop_found_at, score=score, beats=beats, clip=self.slice(start=loop_start, end=loop_found_at)))
        return loops
