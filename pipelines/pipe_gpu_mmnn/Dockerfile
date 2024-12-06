# syntax=docker/dockerfile:1.4
FROM nvidia/cuda:10.2-cudnn7-devel
FROM thoumieupa/train_mmnn:latest
RUN apt-get update && apt-get install -y git

ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
ENV PATH=$CUDA_HOME/bin:$PATH