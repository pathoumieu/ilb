include .env

# Common variables
DOCKER_BUILD = DOCKER_BUILDKIT=1 docker build
DOCKER_TAG = docker tag
DOCKER_PUSH = docker push
DOCKER_RUN = docker run -v /home/pa/projects/ilb/data:/train_ilb/data --shm-size=12g
DOCKER_PS = docker ps -a

# Targets
TARGETS = ilb tabnet mmnn gpu_mmnn

.PHONY: all build push run $(TARGETS)

all: build push run

build: $(TARGETS:%=build-%)

push: $(TARGETS:%=push-%)

run: $(TARGETS:%=run-%)

define build_image
	$(DOCKER_BUILD) \
	--build-arg WANDB_API_KEY=${WANDB_API_KEY} \
	-t thoumieupa/train_$1:latest -f pipe_$1/Dockerfile .
endef

define tag_and_push_image
	$(DOCKER_TAG) thoumieupa/train_$1 thoumieupa/train_$1:latest
	$(DOCKER_PUSH) thoumieupa/train_$1:latest
endef

define run_docker
	$(DOCKER_RUN) thoumieupa/train_$1:latest
endef

define run_docker_with_env
	$(DOCKER_RUN) \
	-e WANDB_PROJECT=${WANDB_PROJECT} \
	-e WANDB_ENTITY=${WANDB_ENTITY} \
	-e WANDB_API_KEY=${WANDB_API_KEY} \
	-e WANDB_DOCKER=thoumieupa/train_$1:latest \
	thoumieupa/train_$1:latest
endef

$(TARGETS):
	@$(MAKE) build-$@
	@$(MAKE) push-$@

build-%:
	$(call build_image,$*)

push-%:
	$(call tag_and_push_image,$*)

run-%:
	$(call run_docker,$*)

job-%:
	$(call run_docker_with_env,$*)

ps:
	$(DOCKER_PS)