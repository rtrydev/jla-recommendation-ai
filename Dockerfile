FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

WORKDIR /src

RUN apt update && apt install -y python3.10 python3.10-venv python3-pip
COPY requirements.txt .
RUN pip3 install -r requirements.txt
RUN python3 -m unidic download