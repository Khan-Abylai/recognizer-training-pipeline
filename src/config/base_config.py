import string

base_folder = '/'

img_w = 160
img_h = 64

img_extensions = ['.jpg', '.png', '.jpeg']
checkpoint_ext = '.pth'
eu_sym = {'å': '@', 'ä': '&', 'ć': '!', 'č': '?', 'đ': '%', 'ö': '^', 'ü': '#', 'š': '$', 'ž': '|'}
eu_add = ''.join(str(val) for key, val in eu_sym.items())
alphabet = string.digits + string.ascii_lowercase + eu_add
num_class = len(alphabet) + 1
# print(num_class, alphabet)
epochs = 500
batch_size = 128
n_cpu = 64
lr = 0.0001

model_lstm_layers = 2
model_lsrm_is_bidirectional = True

model_name = 'eu'
model_extension = '.pth'

checkpoint = ''
checkpoint_dir = '/mnt/sdb1/europe_last/weights'

data_dir = '/europe_last'

######
regions = ['albania', 'andorra', 'austria', 'belgium', 'bosnia', 'bulgaria', 'croatia', 'cyprus', 'czech', 'estonia',
           'finland', 'france', 'germany', 'greece', 'hungary', 'ireland', 'italy', 'latvia',
           'licht', 'lithuania', 'luxemburg', 'makedonia', 'malta', 'monaco', 'montenegro', 'netherlands', 'poland',
           'portugal', 'romania', 'san_marino', 'serbia', 'slovakia', 'slovenia', 'spain', 'sweden', 'swiss']

num_regions = len(regions)

# print("regions", num_regions)
