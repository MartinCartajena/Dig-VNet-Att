
docker build -t vnet_v1 .

# TRAIN COMMAND: train.py

LOCAL
docker run -d --rm --name vnet_v1_train -v "$(pwd):/app" -v "C:/Data/Dig-CS-VNet/Task_LUNA16_test:/app/data" vnet_v1 python /app/main.py --batchSz 1 --dice --nEpochs 300 --opt adam --no-cuda --data_format npy --data_path /app/data/ --infer_data_path /app/data/

CPUHL00003 --> GPU 0
docker run -d --user $(id -u):$(id -g) --rm --name vnet_v1_train_v2 --gpus "device=0" -v $(pwd):/app -v /media/data/mcartajena/LUCIA/Data/LUNA/:/app/data vnet_v1 python /app/main.py --batchSz 1 --dice --nEpochs 300 --loss CE --opt adam --data_format npy --data_path /app/data/ --infer_data_path /app/data/

CPUHL00003 --> NO GPU
docker run --user $(id -u):$(id -g) --rm --name vnet_v1_train_2 -v $(pwd):/app -v /media/data/mcartajena/LUCIA/Data/LUNA/:/app/data vnet_v1 python /app/main.py --batchSz 1 --dice --no-cuda --nEpochs 300 --opt adam --data_format npy --data_path /app/data/ --infer_data_path /app/data/


# TRAIN COMMAND: train_v2.py 

# DEBUG
docker run --rm -it --gpus "device=0" --name vnet_v1_train_v2 -v $(pwd):/app -v /media/data/mcartajena/LUCIA/Data/LUNA/:/app/data vnet_v1 /bin/bash


python /app/train_v2.py --batch_size 16 --epochs 1 --lr 0.0001 --device cuda:0 --workers 4 --vis --data_path /app/data/


# EXECUTION
docker run -d --rm --name vnet_v1_train_v2 --gpus "device=0" -v $(pwd):/app -v /media/data/mcartajena/LUCIA/Data/LUNA16/Task_LUNA16_test:/app/data vnet_v1 python /app/train_v2.py --batch_size 16 --epochs 10 --lr 0.0001 --device cuda:0 --workers 1 --weights ./weights --logs ./logs --image_path /app/data/ --data_format npy










