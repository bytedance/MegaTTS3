# Copyright 2025 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Post-processing regression tests. Needs no checkpoint and no GPU.

Run with `python tests/test_audio_io.py`, or through pytest.
"""

import io
import os
import sys

import numpy as np
import pyloudnorm as pyln
from scipy.io import wavfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tts.utils.audio_utils.io import loudness_norm, to_wav_bytes  # noqa: E402

SR = 24000


def _tone(amplitude, seconds=1.0):
    t = np.arange(int(SR * seconds)) / SR
    return (amplitude * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def test_loudness_norm_keeps_gated_segment_finite():
    """A segment below the BS.1770 absolute gate must not turn into NaN/Inf."""
    meter = pyln.Meter(SR)
    for wav in (np.zeros(SR, dtype=np.float32), _tone(1e-5)):
        loudness = meter.integrated_loudness(wav.astype(float))
        assert not np.isfinite(loudness), 'expected the gated -inf loudness here'
        out = loudness_norm(wav, loudness, -23.0)
        assert np.all(np.isfinite(out)), 'gated segment was destroyed by normalization'
        assert np.array_equal(out, wav), 'gated segment should pass through unchanged'


def test_loudness_norm_matches_previous_behaviour_on_valid_audio():
    """Healthy audio must keep going through the pre-existing code path."""
    meter = pyln.Meter(SR)
    for amplitude in (0.01, 0.3, 0.98, 4.0):
        wav = _tone(amplitude)
        loudness = meter.integrated_loudness(wav.astype(float))
        expected = pyln.normalize.loudness(wav, loudness, -23.0)
        if np.abs(expected).max() >= 1:
            expected = expected / np.abs(expected).max() * 0.95
        np.testing.assert_array_equal(loudness_norm(wav, loudness, -23.0), expected)


def test_to_wav_bytes_rejects_non_finite_audio():
    """Non-finite samples must raise instead of being cast to digital silence."""
    wav = _tone(0.3).astype(float)
    wav[1000:1500] = np.nan
    try:
        to_wav_bytes(wav, SR)
    except ValueError as err:
        assert 'non-finite' in str(err)
    else:
        raise AssertionError('to_wav_bytes silently encoded NaN samples as silence')


def test_to_wav_bytes_still_encodes_valid_audio():
    wav = _tone(0.3).astype(float)
    sr_read, data = wavfile.read(io.BytesIO(to_wav_bytes(wav, SR)))
    assert sr_read == SR
    np.testing.assert_array_equal(data, (wav * 32767).astype(np.int16))


if __name__ == '__main__':
    for _name, _fn in sorted(globals().items()):
        if _name.startswith('test_'):
            _fn()
            print(f'| {_name} ok')
    print('| all tests passed')
