import string

base_folder = '/'

img_w = 160
img_h = 64

img_extensions = ['.jpg', '.png', '.jpeg']
checkpoint_ext = '.pth'
alphabet = string.digits + string.ascii_lowercase
num_class = len(alphabet) + 1

epochs = 500
batch_size = 256
n_cpu = 128
lr = 0.0001

model_lstm_layers = 2
model_lsrm_is_bidirectional = True

model_name = '/model_mena_iter5'
model_extension = '.pth'

checkpoint = ''
checkpoint_dir = '/workspace/data/uae/weights'

data_dir = '/workspace'

######
regions = ["dubai", "abu-dhabi", "sharjah", "ajman", "ras-al-khaimah", "fujairah", "alquwain", "bahrein", "oman", "saudi", "quatar", "kuwait", "others"]
num_regions = len(regions)
