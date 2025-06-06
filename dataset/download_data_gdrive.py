import gdown
import subprocess

url = "https://drive.google.com/drive/folders/1SQRQaJaj_AbQAbjwm8q4efPht-lr3h9J?usp=sharing"
gdown.download_folder(url)


bashCommand = "mv dataset/actual dataset/rb_av ./"
process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
output, error = process.communicate()