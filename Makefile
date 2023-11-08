include .env

# DOCKER TASKS
## Base
build:
	DOCKER_BUILDKIT=1 docker build \
	--build-arg WANDB_API_KEY=${WANDB_API_KEY} \
	-t train_ilb:latest .

push:
	docker tag train_ilb thoumieupa/train_ilb:latest
	docker push thoumieupa/train_ilb:latest

run:
	docker run --volume $(PWD)/:/train_ilb thoumieupa/train_ilb:latest

job:
	docker run -e WANDB_PROJECT=${WANDB_PROJECT} \
	-e WANDB_ENTITY=${WANDB_ENTITY} \
	-e WANDB_API_KEY=${WANDB_API_KEY} \
	-e WANDB_DOCKER=${WANDB_DOCKER} \
	thoumieupa/train_ilb:latest

ilb: build push