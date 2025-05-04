docker run --gpus all --rm -it \
    --name occlusion-container \
    --shm-size=8g \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v "$(pwd)":/workspace \
    occlusion-tracker:v1