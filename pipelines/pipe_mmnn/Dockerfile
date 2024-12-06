# syntax=docker/dockerfile:1.4

FROM pathoumieu/train_tabnet:latest
RUN apt update && apt install gcc -y && apt install -y git

RUN pip install --upgrade pip
COPY --link pipelines/pipe_mmnn/requirements.txt ./
RUN pip install -r requirements.txt

COPY --link pipelines/pipe_mmnn/train_mmnn.py ./
COPY --link utils.py ./
COPY --link utils_torch.py ./
COPY --link models/ ./
COPY --link pipelines/pipe_mmnn/config.yml ./

ENTRYPOINT ["python", "train_mmnn.py"]