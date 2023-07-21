import onnxruntime
import numpy as np
import pdb
import matplotlib.pyplot as plt

ort_session = onnxruntime.InferenceSession("segmentation_resnet.onnx")


img = np.float32(np.expand_dims(np.load('case156_day0_slice_0072.npy').transpose((2, 0, 1)), axis=0))
mx = np.max(img)
if mx:
    img /= mx
print(img.shape)
ort_inputs = {ort_session.get_inputs()[0].name: img}
ort_outs = ort_session.run(None, ort_inputs)

seg = (ort_outs[0][0].transpose((1, 2, 0)) > 0) * 255

plt.subplot(1, 2, 1)
plt.imshow(img[0, 0, :, :])
plt.subplot(1, 2, 2)
plt.imshow(seg)
plt.show()
