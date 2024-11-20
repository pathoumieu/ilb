# syntax=docker/dockerfile:1.4
FROM nvidia/cuda:10.2-cudnn7-devel
FROM thoumieupa/train_mmnn:latest
RUN apt-get update && apt-get install -y git

ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
ENV PATH=$CUDA_HOME/bin:$PATH

RUN pip install --upgrade pip
COPY --link pipe_mmnn/requirements.txt ./
RUN pip install -r requirements.txt

COPY --link pipe_mmnn/train_mmnn.py ./
COPY --link utils.py ./
COPY --link utils_torch.py ./
COPY --link models.py ./
COPY --link pipe_mmnn/config.yml ./

ENTRYPOINT ["python", "train_mmnn.py"]