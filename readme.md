<div align="center">
    <h1>
    MegaTTS 3 <img src="./assets/fig/Hi.gif" width="40px">
    </h1>
    <p>
    Official PyTorch Implementation<br>
    </p>
</div>
<div align="center">
    <a href="https://huggingface.co/spaces/ByteDance/MegaTTS3"><img src="https://img.shields.io/badge/Hugging%20Face-Space%20Demo-yellow" alt="Hugging Face"></a>
    <a href="#"><img src="https://img.shields.io/badge/Platform-linux-lightgrey" alt="version"></a>
    <a href="#"><img src="https://img.shields.io/badge/Python-3.10-brightgreen" alt="version"></a>
    <a href="#"><img src="https://img.shields.io/badge/PyTorch-2.3.0-orange" alt="python"></a>
    <a href="#"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="mit"></a>
</div>
<div align="center">
    <img src="https://img.shields.io/badge/Bytedance-%230077B5.svg?&style=flat-square&logo=bytedance&logoColor=white" />
    <img src="https://img.shields.io/badge/Zhejiang University-%230077B5.svg?&style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA1MTIgNTEyIj48IS0tIUZvbnQgQXdlc29tZSBGcmVlIDYuNy4yIGJ5IEBmb250YXdlc29tZSAtIGh0dHBzOi8vZm9udGF3ZXNvbWUuY29tIExpY2Vuc2UgLSBodHRwczovL2ZvbnRhd2Vzb21lLmNvbS9saWNlbnNlL2ZyZWUgQ29weXJpZ2h0IDIwMjUgRm9udGljb25zLCBJbmMuLS0+PHBhdGggZmlsbD0iI2ZmZmZmZiIgZD0iTTI0My40IDIuNmwtMjI0IDk2Yy0xNCA2LTIxLjggMjEtMTguNyAzNS44UzE2LjggMTYwIDMyIDE2MGwwIDhjMCAxMy4zIDEwLjcgMjQgMjQgMjRsNDAwIDBjMTMuMyAwIDI0LTEwLjcgMjQtMjRsMC04YzE1LjIgMCAyOC4zLTEwLjcgMzEuMy0yNS42cy00LjgtMjkuOS0xOC43LTM1LjhsLTIyNC05NmMtOC0zLjQtMTcuMi0zLjQtMjUuMiAwek0xMjggMjI0bC02NCAwIDAgMTk2LjNjLS42IC4zLTEuMiAuNy0xLjggMS4xbC00OCAzMmMtMTEuNyA3LjgtMTcgMjIuNC0xMi45IDM1LjlTMTcuOSA1MTIgMzIgNTEybDQ0OCAwYzE0LjEgMCAyNi41LTkuMiAzMC42LTIyLjdzLTEuMS0yOC4xLTEyLjktMzUuOWwtNDgtMzJjLS42LS40LTEuMi0uNy0xLjgtMS4xTDQ0OCAyMjRsLTY0IDAgMCAxOTItNDAgMCAwLTE5Mi02NCAwIDAgMTkyLTQ4IDAgMC0xOTItNjQgMCAwIDE5Mi00MCAwIDAtMTkyek0yNTYgNjRhMzIgMzIgMCAxIDEgMCA2NCAzMiAzMiAwIDEgMSAwLTY0eiIvPjwvc3ZnPg==&logoColor=white" />
</div>

## Key features
- 🚀**Lightweight and Efficient:** The backbone of the TTS Diffusion Transformer has only 0.45B parameters.
- 🎧**Ultra High-Quality Voice Cloning:** You can try our model at [Huggingface Demo](https://huggingface.co/spaces/ByteDance/MegaTTS3)🎉. The .wav and .npy files can be found at [link1](https://drive.google.com/drive/folders/1QhcHWcy20JfqWjgqZX1YM3I6i9u4oNlr?usp=sharing). Submit a sample (.wav format, < 24s, and please do not contain space in filename) on [link2](https://drive.google.com/drive/folders/1gCWL1y_2xu9nIFhUX_OW5MbcFuB7J5Cl?usp=sharing) to receive .npy voice latents you can use locally.
- 🌍**Bilingual Support:** Supports both Chinese and English, and code-switching.
- ✍️**Controllable:** Supports accent intensity control ✅ and fine-grained pronunciation/duration adjustment (coming soon).

[MegaTTS 3 Demo Video](https://github.com/user-attachments/assets/0174c111-f392-4376-a34b-0b5b8164aacc)

<div style='width:100%;text-align:center'>
<img src="./assets/fig/table_tts.png" width="550px">
</div>

## 🎯Roadmap

- **[2025-03-22]** Our project has been released!


## Installation
``` sh
# Clone the repository
git clone https://github.com/bytedance/MegaTTS3
cd MegaTTS3
```
**Requirements (for Linux)**
``` sh

# Create a python 3.10 conda env (you could also use virtualenv)
conda create -n megatts3-env python=3.10
conda activate megatts3-env
pip install -r requirements.txt

# Set the root directory
export PYTHONPATH="/path/to/MegaTTS3:$PYTHONPATH"

# [Optional] Set GPU
export CUDA_VISIBLE_DEVICES=0

# If you encounter bugs with pydantic in inference, you should check if the versions of pydantic and gradio are matched.
# [Note] if you encounter bugs related with httpx, please check that whether your environmental variable "no_proxy" has patterns like "::"
```

**Requirements (for Windows)**
``` sh
# [The Windows version is currently under testing]
# Comment below dependence in requirements.txt:
# # WeTextProcessing==1.0.4.1

# Create a python 3.10 conda env (you could also use virtualenv)
conda create -n megatts3-env python=3.10
conda activate megatts3-env
pip install -r requirements.txt
conda install -y -c conda-forge pynini==2.1.5
pip install WeTextProcessing==1.0.3

# [Optional] If you want GPU inference, you may need to install specific version of PyTorch for your GPU from https://pytorch.org/.
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# [Note] if you encounter bugs related with `ffprobe` or `ffmpeg`, you can install it through `conda install -c conda-forge ffmpeg`

# Set environment variable for root directory
set PYTHONPATH="C:\path\to\MegaTTS3;%PYTHONPATH%" # Windows
$env:PYTHONPATH="C:\path\to\MegaTTS3;%PYTHONPATH%" # Powershell on Windows
conda env config vars set PYTHONPATH="C:\path\to\MegaTTS3;%PYTHONPATH%" # For conda users

# [Optional] Set GPU
set CUDA_VISIBLE_DEVICES=0 # Windows
$env:CUDA_VISIBLE_DEVICES=0 # Powershell on Windows

```

**Requirements (for Docker)**
``` sh
# [The Docker version is currently under testing]
# ! You should download the pretrained checkpoint before running the following command
docker build . -t megatts3:latest

# For GPU inference
docker run -it -p 7929:7929 --gpus all -e CUDA_VISIBLE_DEVICES=0 megatts3:latest
# For CPU inference
docker run -it -p 7929:7929  megatts3:latest

# Visit http://0.0.0.0:7929/ for gradio.
```


**Model Download**

The pretrained checkpoint can be found at [Google Drive](https://drive.google.com/drive/folders/1CidiSqtHgJTBDAHQ746_on_YR0boHDYB?usp=sharing) or [Huggingface](https://huggingface.co/ByteDance/MegaTTS3). Please download them and put them to ``./checkpoints/xxx``.

> [!IMPORTANT]  
> For security issues, we do not upload the parameters of WaveVAE encoder to the above links. You can only use the pre-extracted latents from [link1](https://drive.google.com/drive/folders/1QhcHWcy20JfqWjgqZX1YM3I6i9u4oNlr?usp=sharing) for inference. If you want to synthesize speech for speaker A, you need "A.wav" and "A.npy" in the same directory. If you have any questions or suggestions for our model, please email us.
> 
> This project is primarily intended for academic purposes. For academic datasets requiring evaluation, you may upload them to the voice request queue in [link2](https://drive.google.com/drive/folders/1gCWL1y_2xu9nIFhUX_OW5MbcFuB7J5Cl?usp=sharing) (within 24s for each clip). After verifying that your uploaded voices are free from safety issues, we will upload their latent files to [link1](https://drive.google.com/drive/folders/1QhcHWcy20JfqWjgqZX1YM3I6i9u4oNlr?usp=sharing) as soon as possible.
> 
> In the coming days, we will also prepare and release the latent representations for some common TTS benchmarks.

## Inference

**Command-Line Usage (Standard)**
``` bash
# p_w (intelligibility weight), t_w (similarity weight). Typically, prompt with more noises requires higher p_w and t_w
python tts/infer_cli.py --input_wav 'assets/Chinese_prompt.wav'  --input_text "另一边的桌上,一位读书人嗤之以鼻道,'佛子三藏,神子燕小鱼是什么样的人物,李家的那个李子夜如何与他们相提并论？'" --output_dir ./gen

# As long as audio volume and pronunciation are appropriate, increasing --t_w within reasonable ranges (2.0~5.0)
# will increase the generated speech's expressiveness and similarity (especially for some emotional cases).
python tts/infer_cli.py --input_wav 'assets/English_prompt.wav' --input_text 'As his long promised tariff threat turned into reality this week, top human advisers began fielding a wave of calls from business leaders, particularly in the automotive sector, along with lawmakers who were sounding the alarm.' --output_dir ./gen --p_w 2.0 --t_w 3.0
```

> [!TIP]
> You can run `infer_cli_log.py` with a `--logging` flag to track the time spent at each step, such as:
> ```
> python -m tts.infer_cli_log --input_wav 'assets/English_prompt.wav' --input_text 'As his long promised tariff threat turned into reality this week, top human advisers began fielding a wave of calls from business leaders, particularly in the automotive sector, along with lawmakers who were sounding the alarm.' --output_dir ./gen --p_w 2.0 --t_w 3.0 --logging
> ```
> This will show the startpoint and endpoint for each section and store a `timing_logging.log` file.

**Command-Line Usage (for TTS with Accents)**
``` bash
# When p_w (intelligibility weight) ≈ 1.0, the generated audio closely retains the speaker’s original accent. As p_w increases, it shifts toward standard pronunciation. 
# t_w (similarity weight) is typically set 0–3 points higher than p_w for optimal results.
# Useful for accented TTS or solving the accent problems in cross-lingual TTS.
python tts/infer_cli.py --input_wav 'assets/English_prompt.wav' --input_text '这是一条有口音的音频。' --output_dir ./gen --p_w 1.0 --t_w 3.0

python tts/infer_cli.py --input_wav 'assets/English_prompt.wav' --input_text '这条音频的发音标准一些了吗？' --output_dir ./gen --p_w 2.5 --t_w 2.5
```

**Web UI Usage**
``` bash
# We also support cpu inference, but it may take about 30 seconds (for 10 inference steps).
python tts/gradio_api.py
```

## Submodules
> [!TIP]
> In addition to TTS, some submodules in this project may also have additional usages.
> See ``./tts/frontend_fuction.py`` and ``./tts/infer_cli.py`` for example code.

### Aligner
**Description:** a robust speech-text aligner model trained using pseudo-labels generated by a large number of MFA expert models.

**Usage**: 1) Prepare the finetuning dataset for our model; 2) Filter the large-scale speech dataset (if the aligner fails to align a certain speech clip, it is likely to be noisy); 3) Phoneme recognition; 4) Speech segmentation.

### Graphme-to-Phoneme Model
**Description:** a Qwen2.5-0.5B model finetuned for robust graphme-to-phoneme conversion.

**Usage**: Graphme-to-phoneme conversion.

### WaveVAE
**Description:** a strong waveform VAE that can compress 24 kHz speeche into 25 Hz acoustic latent and reconstruct the original wave almost losslessly.

**Usage:** 1) Acoustic latents can provide a more compact and discriminative training target for speech synthesis models compared to mel-spectrograms, accelerating convergence; 2) Used as acoustic latents for voice conversion; 3) High-quality vocoder.

<div style='width:100%;text-align:center'>
<img src="./assets/fig/table_wavvae.png" width="650px">
</div>


## Security
If you discover a potential security issue in this project, or think you may
have discovered a security issue, we ask that you notify Bytedance Security via our [security center](https://security.bytedance.com/src) or [sec@bytedance.com](sec@bytedance.com).

Please do **not** create a public GitHub issue.

## License
This project is licensed under the [Apache-2.0 License](LICENSE).

## Citation
This repo contains forced-align version of `Sparse Alignment Enhanced Latent Diffusion Transformer for Zero-Shot Speech Synthesis` and the WavVAE is mainly based on `Wavtokenizer: an efficient acoustic discrete codec tokenizer for audio language modeling`. Compared to the model described in paper, the repository includes additional models. These models not only enhance the stability and cloning capabilities of the algorithm but can also be independently utilized to serve a wider range of scenarios.
```
@article{jiang2025sparse,
  title={Sparse Alignment Enhanced Latent Diffusion Transformer for Zero-Shot Speech Synthesis},
  author={Jiang, Ziyue and Ren, Yi and Li, Ruiqi and Ji, Shengpeng and Ye, Zhenhui and Zhang, Chen and Jionghao, Bai and Yang, Xiaoda and Zuo, Jialong and Zhang, Yu and others},
  journal={arXiv preprint arXiv:2502.18924},
  year={2025}
}

@article{ji2024wavtokenizer,
  title={Wavtokenizer: an efficient acoustic discrete codec tokenizer for audio language modeling},
  author={Ji, Shengpeng and Jiang, Ziyue and Wang, Wen and Chen, Yifu and Fang, Minghui and Zuo, Jialong and Yang, Qian and Cheng, Xize and Wang, Zehan and Li, Ruiqi and others},
  journal={arXiv preprint arXiv:2408.16532},
  year={2024}
}
```
