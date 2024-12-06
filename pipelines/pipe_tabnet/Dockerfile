# syntax=docker/dockerfile:1.4

FROM pathoumieu/train_ilb:latest
RUN apt update && apt install gcc -y && apt install -y git

RUN pip install --upgrade pip
COPY --link pipelines/pipe_tabnet/requirements.txt ./
RUN pip install -r requirements.txt

COPY --link pipelines/pipe_tabnet/train_tabnet.py ./
COPY --link utils.py ./
COPY --link utils_torch.py ./
COPY --link pipelines/pipe_tabnet/config.yml ./

ENTRYPOINT ["python", "train_tabnet.py"]