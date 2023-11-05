# syntax=docker/dockerfile:1.4

FROM thoumieupa/train_base:latest
RUN apt update && apt install gcc -y && apt install -y git

WORKDIR /train_ilb

COPY --link train_ilb.py ./

RUN pip install --upgrade pip
COPY --link requirements.txt ./
RUN pip install -r requirements.txt

COPY --link data/X_train_J01Z4CN.csv ./
COPY --link data/y_train_OXxrJt1.csv ./
COPY --link data/X_test_BEhvxAN.csv ./
COPY --link data/y_random_MhJDhKK.csv ./

ENTRYPOINT ["python", "train_ilb.py"]