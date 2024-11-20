# syntax=docker/dockerfile:1.4

FROM thoumieupa/train_tabnet:latest
RUN apt update && apt install gcc -y && apt install -y git

COPY --link pipe_tabnet_pretraining/train_tabnet_pretraining.py ./

ENTRYPOINT ["python", "train_tabnet_pretraining.py"]