run:
	docker run -d -it --ipc=host --gpus all -v /mnt/storage:/mnt --cap-add sys_ptrace \
		-p127.0.0.1:5555:22 --name application registry.infra.smartparking.kz/nomeroff:dev