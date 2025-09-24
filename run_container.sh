#!/usr/bin/env sh


# Project directory
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

# Run the container interactively
# With mounted volumes
# in the background to allow mutliple tmux
# panes attaching to the same session
docker run -dit \
	-e TERM="$TERM" \
	-v "$PROJECT_DIR:/workspace" \
	-v "$HOME/.vim:/root/.vim" \
	-v "$HOME/.vimrc:/root/.vimrc" \
	--name "$CONTAINER" \
	"$IMAGE"

# Attach to the container
docker exec -it "$CONTAINER" zsh
