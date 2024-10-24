FROM python:3.9.0-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# docker build -t vnet_v1 .

# docker run -d --rm -it --name vnet_v1_train -v "$(pwd):/app" -v "C:/Data/Dig-CS-VNet/Task_LUNA16_test:/app/data" vnet_v1 python /app/main.py --batchSz 1 --dice --nEpochs 300 --opt adam --no-cuda --data_format npy --data_path /app/data/ --infer_data_path /app/data/
