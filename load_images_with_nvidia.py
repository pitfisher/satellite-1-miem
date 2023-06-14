import time
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from nvidia.dali.backend import TensorListGPU

image_dir = "data/test_data/images"
seed = 1549361629
batch_size = 100

@pipeline_def(seed=seed)
def image_decoder_pipeline(device="cpu"):
    jpegs, labels = fn.readers.file(file_root=image_dir)
    return fn.decoders.image(jpegs, device=device)

pipe = image_decoder_pipeline(batch_size=batch_size, num_threads=1, device_id=0)

images_loading_time = 0
image_loading_start_time = time.perf_counter()

pipe.build()
images, = pipe.run()
# if isinstance(images, TensorListGPU):
#     images = images.as_cpu()

image_loading_time_elapsed = time.perf_counter() - image_loading_start_time
print("image loaded, time elapsed: {:.3f}".format(image_loading_time_elapsed))
print(len(images))
exit()