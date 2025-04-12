@echo off
:: Navigate to Anaconda installation directory
call "D:\Anaconda3\Scripts\activate.bat"
:: Activate environment
call conda activate megatts3-env
:: Change to E drive
e:
:: Navigate to MegaTTS3 directory
cd E:\APPs\MegaTTS3
:: Run the Python script
python -m tts.gradio_api
:: Keep window open after execution (optional)
pause