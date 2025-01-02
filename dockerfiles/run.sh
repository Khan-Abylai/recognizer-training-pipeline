docker run -d -it --rm --ipc=host --cap-add sys_ptrace -p0.0.0.0:3030:22 \
            --gpus all \
            --name lp_recognizer \
            doc.smartparking.kz/lp_recognizer:1.0
