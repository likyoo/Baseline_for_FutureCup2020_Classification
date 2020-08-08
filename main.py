import os
from naie.datasets import get_data_reference
import moxing as mox
from naie.context import Context
import zipfile

os.system('nvidia-smi')
os.system('ls')

os.system('python -m pip install --upgrade pip')
os.system('pip install tensorboardX tensorboard')

data_reference = get_data_reference(dataset="DatasetService", dataset_entity='OceanDataset')
for file_paths in data_reference.get_files_paths():
    mox.file.copy(file_paths, '/cache/' + file_paths.split('/')[-1])
zip_file = zipfile.ZipFile('/cache/data.zip')
zip_list = zip_file.namelist()
for f in zip_list:
    zip_file.extract(f, '/cache/')

#  开发样本
mox.file.copy('annotation.csv', '/cache/annotation.csv')
os.system('ls /cache')
#

os.system('python train.py')

mox.file.copy_parallel('runs', os.path.join(Context.get_output_path(), 'runs'))
mox.file.copy_parallel('checkpoint', os.path.join(Context.get_output_path(), 'checkpoint'))
print(os.path.join(Context.get_output_path() + '/runs'))

os.system('python test.py')
mox.file.copy('ans.csv', os.path.join(Context.get_output_path(), 'ans.csv'))
