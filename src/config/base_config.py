import string

base_folder = '/'

img_w = 128
img_h = 32

img_extensions = ['.jpg', '.png', '.jpeg']
checkpoint_ext = '.pth'
alphabet = string.digits + string.ascii_lowercase + '.'
num_class = len(alphabet) + 1

epochs = 100
batch_size = 1024
n_cpu = 16
lr = 0.0001

model_lstm_layers = 1
model_lsrm_is_bidirectional = False

model_name = 'tmp'
model_extension = '.pth'

checkpoint = ''
checkpoint_dir = ''

data_dir = ''

######
regions = ["dubai", "abu-dhabi", "sharjah", "ajman", "ras-al-khaimah", "fujairah", "alquwain"]
num_regions = len(regions)
