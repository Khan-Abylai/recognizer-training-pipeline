docker run -d -it --rm --ipc=host --cap-add sys_ptrace -p0.0.0.0:3030:22 \
            -v /run/media/artykbayevk/51ee93fe-2e3e-4647-84bb-6ee320319dc6/recognizer_mount:/home/user/data \
            --gpus all \
            --name lp_recognizer \
            doc.smartparking.kz/lp_recognizer:1.0
