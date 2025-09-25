#!/usr/bin/env sh

set -eu

# oProject directory
# Should not be root
PROJECT_DIR="$PWD"

# Project name
# Assumes the name of the project
# is the name of the pwd
PROJECT="${PWD##*/}"

# Image name
IMAGE="${PROJECT}-venv"

# Container name
CONTAINER="${PROJECT}-container"


# If there's no requirements.txt, make an empty one
if [ ! -f "requirements.txt" ]; then
	touch requirements.txt
fi

# If no dockerfile, make a template
if [ ! -f "Dockerfile" ]; then
	cp ~/Dockerfile_template.txt Dockerfile.dev
	cat Dockerfile.dev
	echo
	echo "This is what the Dockerfile will look like, do you want to make any changes?"
	echo "Respond with y/n"
	echo "If you decide you do want to change, modify Dockerfile.dev with your favorite text editor"
	echo
	read response
	if [ "$response" = "y" ]; then
		exit 0
	fi
fi

# If the image doesn't exits build it
if ! docker image inspect "$IMAGE" > /dev/null 2>&1; then
	docker build -t "$IMAGE" -f Dockerfile.dev .
fi

# Find out if the container exists and is running. If not make one/ start it
if docker container inspect "$CONTAINER" > /dev/null 2>&1; then
	running="$(docker container inspect "$CONTAINER" --format '{{.State.Running}}')"
	if [ "$running" = "false" ]; then
		docker start "$CONTAINER"
	fi
else
	docker run -dit \
		-e TERM="$TERM" \
		-v "$PROJECT_DIR:/workspace" \
		-v "$HOME/.vim:/root/.vim" \
		-v "$HOME/.vimrc:/root/.vimrc" \
		--name "$CONTAINER" \
		"$IMAGE"
fi

# Attach to the container
docker exec -it "$CONTAINER" zsh
