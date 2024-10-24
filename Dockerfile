FROM python:3.9.0-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# docker build -t vnet_v1 .

# docker run -d --user $(id -u):$(id -g) --rm -it --name vnet_v1_train --gpus "device=0" -v $(pwd):/app -v /media/data/mcartajena/LUCIA/Data/LUNA/:/app/data vnet_v1 python /app/main.py --batchSz 1 --dice --nEpochs 300 --opt adam --data_format npy --data_path /app/data/ --infer_data_path /app/data/