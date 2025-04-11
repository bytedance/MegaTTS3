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

import json
import os
import argparse
import librosa
import numpy as np
import torch
import time
import logging
from contextlib import contextmanager

from tn.chinese.normalizer import Normalizer as ZhNormalizer
from tn.english.normalizer import Normalizer as EnNormalizer
from langdetect import detect as classify_language
from pydub import AudioSegment
import pyloudnorm as pyln

from tts.modules.ar_dur.commons.nar_tts_modules import LengthRegulator
from tts.frontend_function import g2p, align, make_dur_prompt, dur_pred, prepare_inputs_for_dit
from tts.utils.audio_utils.io import save_wav, to_wav_bytes, convert_to_wav_bytes, combine_audio_segments
from tts.utils.commons.ckpt_utils import load_ckpt
from tts.utils.commons.hparams import set_hparams, hparams
from tts.utils.text_utils.text_encoder import TokenTextEncoder
from tts.utils.text_utils.split_text import chunk_text_chinese, chunk_text_english, chunk_text_chinesev2
from tts.utils.commons.hparams import hparams, set_hparams

# Performance tracking setup
class PerformanceTracker:
    """Track performance metrics for MegaTTS inference."""
    
    def __init__(self):
        self.metrics = {}
        self.start_time = None
        self.current_section = None
    
    def start_section(self, section_name):
        """Start timing a specific section."""
        if self.current_section:
            logging.warning(f"Starting {section_name} while {self.current_section} is still running")
        
        self.current_section = section_name
        self.start_time = time.time()
        logging.info(f"Starting: {section_name}")
        
    def end_section(self, section_name=None):
        """End timing a specific section and record the time."""
        if section_name is None:
            section_name = self.current_section
            
        if section_name != self.current_section:
            logging.warning(f"Ending {section_name} but {self.current_section} is currently running")
            
        duration = time.time() - self.start_time
        
        if section_name not in self.metrics:
            self.metrics[section_name] = []
        
        self.metrics[section_name].append(duration)
        logging.info(f"Completed: {section_name} - {duration:.4f}s")
        self.current_section = None
        
        return duration
    
    def get_metrics(self):
        """Get recorded metrics."""
        result = {}
        for section, times in self.metrics.items():
            result[section] = {
                "times": times,
                "mean": sum(times) / len(times) if times else 0,
                "min": min(times) if times else 0,
                "max": max(times) if times else 0,
                "count": len(times)
            }
        return result
    
    def log_summary(self):
        """Log a summary of all metrics."""
        metrics = self.get_metrics()
        logging.info("=== Performance Summary ===")
        
        for section, stats in metrics.items():
            logging.info(f"{section}: ")
            logging.info(f"  Count: {stats['count']}")
            logging.info(f"  Mean: {stats['mean']:.4f}s")
            logging.info(f"  Min: {stats['min']:.4f}s")
            logging.info(f"  Max: {stats['max']:.4f}s")
        
        logging.info("=========================")

    @contextmanager
    def track(self, section_name):
        """Context manager for tracking a code section."""
        self.start_section(section_name)
        try:
            yield
        finally:
            self.end_section(section_name)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("tts_performance.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("MegaTTS-Perf")

if "TOKENIZERS_PARALLELISM" not in os.environ:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

def convert_to_wav(wav_path):
    # Check if the file exists
    if not os.path.exists(wav_path):
        print(f"The file '{wav_path}' does not exist.")
        return

    # Check if the file already has a .wav extension
    if not wav_path.endswith(".wav"):
        # Define the output path with a .wav extension
        out_path = os.path.splitext(wav_path)[0] + ".wav"

        # Load the audio file using pydub and convert it to WAV
        audio = AudioSegment.from_file(wav_path)
        audio.export(out_path, format="wav")

        print(f"Converted '{wav_path}' to '{out_path}'")


def cut_wav(wav_path, max_len=28):
    audio = AudioSegment.from_file(wav_path)
    audio = audio[:int(max_len * 1000)]
    audio.export(wav_path, format="wav")

class MegaTTS3DiTInfer():
    def __init__(
            self, 
            device=None,
            ckpt_root='./checkpoints',
            dit_exp_name='diffusion_transformer',
            frontend_exp_name='aligner_lm',
            wavvae_exp_name='wavvae',
            dur_ckpt_path='duration_lm',
            g2p_exp_name='g2p',
            precision=torch.float16,
            enable_profiling=True,
            **kwargs
        ):
        self.sr = 24000
        self.fm = 8
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.precision = precision
        
        # Initialize performance tracker
        self.enable_profiling = enable_profiling
        self.perf_tracker = PerformanceTracker() if enable_profiling else None

        # build models
        with self._track("model_initialization"):
            self.dit_exp_name = os.path.join(ckpt_root, dit_exp_name)
            self.frontend_exp_name = os.path.join(ckpt_root, frontend_exp_name)
            self.wavvae_exp_name = os.path.join(ckpt_root, wavvae_exp_name)
            self.dur_exp_name = os.path.join(ckpt_root, dur_ckpt_path)
            self.g2p_exp_name = os.path.join(ckpt_root, g2p_exp_name)
            self.build_model(self.device)

        # init text normalizer
        self.zh_normalizer = ZhNormalizer(overwrite_cache=False, remove_erhua=False, remove_interjections=False)
        self.en_normalizer = EnNormalizer(overwrite_cache=False)
        # loudness meter
        self.loudness_meter = pyln.Meter(self.sr)

    def _track(self, section_name):
        """Helper method to use the performance tracker."""
        if self.enable_profiling and self.perf_tracker:
            return self.perf_tracker.track(section_name)
        else:
            # Return a no-op context manager
            @contextmanager
            def noop_context():
                yield
            return noop_context()

    def build_model(self, device):
        with self._track("set_hparams"):
            set_hparams(exp_name=self.dit_exp_name, print_hparams=False)

        ''' Load Dict '''
        with self._track("load_dict"):
            current_dir = os.path.dirname(os.path.abspath(__file__))
            ling_dict = json.load(open(f"{current_dir}/utils/text_utils/dict.json", encoding='utf-8-sig'))
            self.ling_dict = {k: TokenTextEncoder(None, vocab_list=ling_dict[k], replace_oov='<UNK>') for k in ['phone', 'tone']}
            self.token_encoder = token_encoder = self.ling_dict['phone']
            ph_dict_size = len(token_encoder)

        ''' Load Duration LM '''
        with self._track("load_duration_model"):
            from tts.modules.ar_dur.ar_dur_predictor import ARDurPredictor
            hp_dur_model = self.hp_dur_model = set_hparams(f'{self.dur_exp_name}/config.yaml', global_hparams=False)
            hp_dur_model['frames_multiple'] = hparams['frames_multiple']
            self.dur_model = ARDurPredictor(
                hp_dur_model, hp_dur_model['dur_txt_hs'], hp_dur_model['dur_model_hidden_size'],
                hp_dur_model['dur_model_layers'], ph_dict_size,
                hp_dur_model['dur_code_size'],
                use_rot_embed=hp_dur_model.get('use_rot_embed', False))
            self.length_regulator = LengthRegulator()
            load_ckpt(self.dur_model, f'{self.dur_exp_name}', 'dur_model')
            self.dur_model.eval()
            self.dur_model.to(device)

        ''' Load Diffusion Transformer '''
        with self._track("load_dit"):
            from tts.modules.llm_dit.dit import Diffusion
            self.dit = Diffusion()
            load_ckpt(self.dit, f'{self.dit_exp_name}', 'dit', strict=False)
            self.dit.eval()
            self.dit.to(device)
            self.cfg_mask_token_phone = 302 - 1
            self.cfg_mask_token_tone = 32 - 1

        ''' Load Frontend LM '''
        with self._track("load_frontend_lm"):
            from tts.modules.aligner.whisper_small import Whisper
            self.aligner_lm = Whisper()
            load_ckpt(self.aligner_lm, f'{self.frontend_exp_name}', 'model')
            self.aligner_lm.eval()
            self.aligner_lm.to(device)
            self.kv_cache = None
            self.hooks = None

        ''' Load G2P LM'''
        with self._track("load_g2p"):
            from transformers import AutoTokenizer, AutoModelForCausalLM
            g2p_tokenizer = AutoTokenizer.from_pretrained(self.g2p_exp_name, padding_side="right")
            g2p_tokenizer.padding_side = "right"
            self.g2p_model = AutoModelForCausalLM.from_pretrained(self.g2p_exp_name).eval().to(device)
            self.g2p_tokenizer = g2p_tokenizer
            self.speech_start_idx = g2p_tokenizer.encode('<Reserved_TTS_0>')[0]

        ''' Wav VAE '''
        with self._track("load_wavvae"):
            self.hp_wavvae = hp_wavvae = set_hparams(f'{self.wavvae_exp_name}/config.yaml', global_hparams=False)
            from tts.modules.wavvae.decoder.wavvae_v3 import WavVAE_V3
            self.wavvae = WavVAE_V3(hparams=hp_wavvae)
            if os.path.exists(f'{self.wavvae_exp_name}/model_only_last.ckpt'):
                load_ckpt(self.wavvae, f'{self.wavvae_exp_name}/model_only_last.ckpt', 'model_gen', strict=True)
                self.has_vae_encoder = True
            else:
                load_ckpt(self.wavvae, f'{self.wavvae_exp_name}/decoder.ckpt', 'model_gen', strict=False)
                self.has_vae_encoder = False
            self.wavvae.eval()
            self.wavvae.to(device)
            self.vae_stride = hp_wavvae.get('vae_stride', 4)
            self.hop_size = hp_wavvae.get('hop_size', 4)
    
    def preprocess(self, audio_bytes, latent_file=None, topk_dur=1, **kwargs):
        with self._track("preprocess_total"):
            with self._track("load_wav"):
                wav_bytes = convert_to_wav_bytes(audio_bytes)
                ''' Load wav '''
                wav, _ = librosa.core.load(wav_bytes, sr=self.sr)
                # Pad wav if necessary
                ws = hparams['win_size']
                if len(wav) % ws < ws - 1:
                    wav = np.pad(wav, (0, ws - 1 - (len(wav) % ws)), mode='constant', constant_values=0.0).astype(np.float32)
                wav = np.pad(wav, (0, 12000), mode='constant', constant_values=0.0).astype(np.float32)
                self.loudness_prompt = self.loudness_meter.integrated_loudness(wav.astype(float))

            with self._track("align_wav"):
                ''' obtain alignments with aligner_lm '''
                ph_ref, tone_ref, mel2ph_ref = align(self, wav)

            with torch.inference_mode():
                with self._track("vae_encode"):
                    ''' Forward WaveVAE to obtain: prompt latent '''
                    if self.has_vae_encoder:
                        wav = torch.FloatTensor(wav)[None].to(self.device)
                        vae_latent = self.wavvae.encode_latent(wav)
                        vae_latent = vae_latent[:, :mel2ph_ref.size(1)//4]
                    else:
                        assert latent_file is not None, "Please provide latent_file in WaveVAE decoder-only mode"
                        vae_latent = torch.from_numpy(np.load(latent_file)).to(self.device)
                        vae_latent = vae_latent[:, :mel2ph_ref.size(1)//4]
            
                with self._track("duration_prompt"):
                    ''' Duration Prompting '''
                    self.dur_model.hparams["infer_top_k"] = topk_dur if topk_dur > 1 else None
                    incremental_state_dur_prompt, ctx_dur_tokens = make_dur_prompt(self, mel2ph_ref, ph_ref, tone_ref)
            
            return {
                'ph_ref': ph_ref,
                'tone_ref': tone_ref,
                'mel2ph_ref': mel2ph_ref,
                'vae_latent': vae_latent,
                'incremental_state_dur_prompt': incremental_state_dur_prompt,
                'ctx_dur_tokens': ctx_dur_tokens,
            }

    def forward(self, resource_context, input_text, time_step, p_w, t_w, dur_disturb=0.1, dur_alpha=1.0, **kwargs):
        with self._track("forward_total"):
            device = self.device

            with self._track("load_resource_context"):
                ph_ref = resource_context['ph_ref'].to(device)
                tone_ref = resource_context['tone_ref'].to(device)
                mel2ph_ref = resource_context['mel2ph_ref'].to(device)
                vae_latent = resource_context['vae_latent'].to(device)
                ctx_dur_tokens = resource_context['ctx_dur_tokens'].to(device)
                incremental_state_dur_prompt = resource_context['incremental_state_dur_prompt']

            with torch.inference_mode():
                ''' Generating '''
                wav_pred_ = []
                with self._track("text_normalization"):
                    language_type = classify_language(input_text)
                    if language_type == 'en':
                        input_text = self.en_normalizer.normalize(input_text)
                        text_segs = chunk_text_english(input_text, max_chars=130)
                    else:
                        input_text = self.zh_normalizer.normalize(input_text)
                        text_segs = chunk_text_chinesev2(input_text, limit=60)

                all_seg_times = []
                for seg_i, text in enumerate(text_segs):
                    with self._track(f"segment_{seg_i}_total"):
                        segment_start = time.time()
                        
                        with self._track(f"segment_{seg_i}_g2p"):
                            ''' G2P '''
                            ph_pred, tone_pred = g2p(self, text)

                        with self._track(f"segment_{seg_i}_duration"):
                            ''' Duration Prediction '''
                            mel2ph_pred = dur_pred(self, ctx_dur_tokens, incremental_state_dur_prompt, ph_pred, tone_pred, seg_i, dur_disturb, dur_alpha, is_first=seg_i==0, is_final=seg_i==len(text_segs)-1)
                        
                        with self._track(f"segment_{seg_i}_prepare_inputs"):
                            inputs = prepare_inputs_for_dit(self, mel2ph_ref, mel2ph_pred, ph_ref, tone_ref, ph_pred, tone_pred, vae_latent)
                        
                        with self._track(f"segment_{seg_i}_dit_inference"):
                            # Speech dit inference
                            with torch.cuda.amp.autocast(dtype=self.precision, enabled=True):
                                x = self.dit.inference(inputs, timesteps=time_step, seq_cfg_w=[p_w, t_w]).float()
                        
                        with self._track(f"segment_{seg_i}_wavvae_decode"):
                            # WavVAE decode
                            x[:, :vae_latent.size(1)] = vae_latent
                            wav_pred = self.wavvae.decode(x)[0,0].to(torch.float32)
                        
                        with self._track(f"segment_{seg_i}_postprocess"):
                            ''' Post-processing '''
                            # Trim prompt wav
                            wav_pred = wav_pred[vae_latent.size(1)*self.vae_stride*self.hop_size:].cpu().numpy()
                            # Norm generated wav to prompt wav's level
                            meter = pyln.Meter(self.sr)  # create BS.1770 meter
                            loudness_pred = self.loudness_meter.integrated_loudness(wav_pred.astype(float))
                            wav_pred = pyln.normalize.loudness(wav_pred, loudness_pred, self.loudness_prompt)
                            if np.abs(wav_pred).max() >= 1:
                                wav_pred = wav_pred / np.abs(wav_pred).max() * 0.95

                            # Apply hamming window
                            wav_pred_.append(wav_pred)
                            
                        segment_time = time.time() - segment_start
                        all_seg_times.append(segment_time)
                        logging.info(f"Segment {seg_i} processed in {segment_time:.4f}s")

                with self._track("combine_segments"):
                    wav_pred = combine_audio_segments(wav_pred_, sr=self.sr).astype(float)
                
                avg_segment_time = sum(all_seg_times)/len(all_seg_times) if all_seg_times else 0
                logging.info(f"Average segment processing time: {avg_segment_time:.4f}s")
                
                return to_wav_bytes(wav_pred, self.sr)

    def log_performance_summary(self):
        """Log a summary of performance metrics."""
        if self.enable_profiling and self.perf_tracker:
            self.perf_tracker.log_summary()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_wav', type=str)
    parser.add_argument('--input_text', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--time_step', type=int, default=32, help='Inference steps of Diffusion Transformer')
    parser.add_argument('--p_w', type=float, default=1.6, help='Intelligibility Weight')
    parser.add_argument('--t_w', type=float, default=2.5, help='Similarity Weight')
    parser.add_argument('--profile', action='store_true', help='Enable performance profiling')
    args = parser.parse_args()
    wav_path, input_text, out_path, time_step, p_w, t_w = args.input_wav, args.input_text, args.output_dir, args.time_step, args.p_w, args.t_w

    # Overall timing
    start_time = time.time()
    
    infer_ins = MegaTTS3DiTInfer(enable_profiling=args.profile)

    with open(wav_path, 'rb') as file:
        file_content = file.read()

    print(f"| Start processing {wav_path}+{input_text}")
    
    preprocess_start = time.time()
    resource_context = infer_ins.preprocess(file_content, latent_file=wav_path.replace('.wav', '.npy'))
    preprocess_time = time.time() - preprocess_start
    print(f"| Preprocessing completed in {preprocess_time:.2f}s")
    
    inference_start = time.time()
    wav_bytes = infer_ins.forward(resource_context, input_text, time_step=time_step, p_w=p_w, t_w=t_w)
    inference_time = time.time() - inference_start
    print(f"| Inference completed in {inference_time:.2f}s")

    print(f"| Saving results to {out_path}/[P]{input_text[:20]}.wav")
    os.makedirs(out_path, exist_ok=True)
    save_wav(wav_bytes, f'{out_path}/[P]{input_text[:20]}.wav')
    
    total_time = time.time() - start_time
    print(f"| Total processing time: {total_time:.2f}s")
    
    # Log performance summary if profiling is enabled
    if args.profile:
        infer_ins.log_performance_summary()