# Include environment variables from .env file
include .env

# Common Docker commands
DOCKER_BUILD = DOCKER_BUILDKIT=1 docker build
DOCKER_TAG = docker tag
DOCKER_PUSH = docker push
DOCKER_RUN = docker run -v ${PWD}/data:/train_ilb/data --shm-size=12g
DOCKER_PS = docker ps -a

# List of pipeline targets
TARGETS = ilb tabnet mmnn gpu_mmnn

# Default make target (build, push, and run all pipelines)
.PHONY: all build push run $(TARGETS)

all: build push run

# Build all pipelines
build: $(TARGETS:%=build-%)

# Push all pipelines
push: $(TARGETS:%=push-%)

# Run all pipelines
run: $(TARGETS:%=run-%)

# Macro to build Docker images for each pipeline
define build_image
	$(DOCKER_BUILD) \
	--build-arg WANDB_API_KEY=${WANDB_API_KEY} \
	-t pathoumieu/train_$1:latest -f pipelines/pipe_$1/Dockerfile .
endef

# Macro to tag and push Docker images for each pipeline
define tag_and_push_image
	$(DOCKER_TAG) pathoumieu/train_$1 pathoumieu/train_$1:latest
	$(DOCKER_PUSH) pathoumieu/train_$1:latest
endef

# Macro to run Docker containers for each pipeline
define run_docker
	$(DOCKER_RUN) pathoumieu/train_$1:latest
endef

# Macro to run Docker containers with environment variables
define run_docker_with_env
	$(DOCKER_RUN) \
	-e WANDB_PROJECT=${WANDB_PROJECT} \
	-e WANDB_ENTITY=${WANDB_ENTITY} \
	-e WANDB_API_KEY=${WANDB_API_KEY} \
	-e WANDB_DOCKER=pathoumieu/train_$1:latest \
	pathoumieu/train_$1:latest
endef

# Make rules for each pipeline: build, push, run
$(TARGETS):
	@$(MAKE) build-$@
	@$(MAKE) push-$@
	@$(MAKE) run-$@

# Rule to build Docker image for a specific pipeline
build-%:
	$(call build_image,$*)

# Rule to tag and push Docker image for a specific pipeline
push-%:
	$(call tag_and_push_image,$*)

# Rule to run Docker container for a specific pipeline
run-%:
	$(call run_docker,$*)

# Rule to run Docker container with environment variables for a specific pipeline
job-%:
	$(call run_docker_with_env,$*)

# List running Docker containers
ps:
	$(DOCKER_PS)