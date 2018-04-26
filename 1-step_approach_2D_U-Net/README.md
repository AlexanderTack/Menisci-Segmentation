# Menisci-Segmentation 1-step approach

The "1-step approach" is based on a 2D U-Net. This 2D U-Net is employed for the segmentation of sagittal weDESS MRI slices from the OAI database (https://oai.epi-ucsf.org/datarelease/).

The 2D U-Net was trained on manual segmentations provided by Imorphics (Manchester, UK).
These manual gold standard segmentations are publicly available as part of the OAI database (https://oai.epi-ucsf.org/datarelease/iMorphics.asp).
All networks were trained on the baseline segmentation masks only.

Weights are supplied at https://opus4.kobv.de/opus4-zib/frontdoor/index/index/docId/6748 for the following networks.
Please note that it can take up to several minutes until the download is starting.

    2D U-Net for the segmentation of medial *and* lateral menisci from sagittal DESS MRIs (trained on the first 44 subjects of the Imorphics data; sorted numerically)
    "menisci-2D-UNet.384x384.TrainedOnFirst44.hdf5"

    2D U-Net for the segmentation of medial *and* lateral menisci from sagittal DESS MRIs (trained on the last 44 subjects of the Imorphics data; sorted numerically)
    "menisci-2D-UNet.384x384.TrainedOnLast44.hdf5"


The code was developed using the following configuration:

    Ubuntu 14.04
    GeForce GTX 1080 Ti
    NVIDIA CUDA Toolkit 8.0
    cuDNN 5.1

    Theano (0.9.0)
    Keras (2.0.2)
    numpy (1.13.3)
    pydicom (0.9.9)

keras.json:

    "image_dim_ordering": "th", 
    "epsilon": 1e-07, 
    "floatx": "float32", 
    "image_data_format": "channels_first", 
    "backend": "theano"
