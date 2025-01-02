import string

base_folder = '/'

img_w = 160
img_h = 64

img_extensions = ['.jpg', '.png', '.jpeg']
checkpoint_ext = '.pth'
alphabet = string.digits + string.ascii_lowercase
num_class = len(alphabet) + 1

epochs = 500
batch_size = 128
n_cpu = 16
lr = 0.0001

model_lstm_layers = 2
model_lsrm_is_bidirectional = True

model_name = 'tmp'
model_extension = '.pth'

checkpoint = ''
checkpoint_dir = '/home/user/data/experiment/fine_tuning_for_UAE/base_model_weights'

data_dir = '/home/user/data'

######
regions = ["dubai", "abu-dhabi", "sharjah", "ajman", "ras-al-khaimah", "fujairah", "alquwain"]
num_regions = len(regions)
