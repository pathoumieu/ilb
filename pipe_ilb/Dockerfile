# syntax=docker/dockerfile:1.4

FROM thoumieupa/train_base:latest
RUN apt update && apt install gcc -y && apt install -y git

RUN pip install --upgrade pip
COPY --link pipe_ilb/requirements.txt ./
RUN pip install -r requirements.txt

WORKDIR /train_ilb
RUN mkdir /train_ilb/data

ARG WANDB_API_KEY
ENV WANDB_API_KEY=${WANDB_API_KEY}
ENV CONFIG_FILE_DIR=.
ENV DATA_FILE_DIR=./data


COPY --link pipe_ilb/train_ilb.py ./
COPY --link utils.py ./
COPY --link pipe_ilb/config.yml ./

COPY --link data/X_train_J01Z4CN.csv ./data/
COPY --link data/y_train_OXxrJt1.csv ./data/
COPY --link data/X_test_BEhvxAN.csv ./data/
COPY --link data/y_random_MhJDhKK.csv ./data/

ENTRYPOINT ["python", "train_ilb.py"]