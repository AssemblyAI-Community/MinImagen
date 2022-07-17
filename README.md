# MinImagen
### A Minimal implementation of the [Imagen](https://imagen.research.google/) text-to-image model.

![Imagen model structure](./image/model_structure.png)

Given a caption of an image, Imagen will generate an image that reflects the caption. The model is a simple [cascading diffusion model](https://arxiv.org/abs/2106.15282), using a T5 text encoder to encode the captions which conditions a base image generator, and then a sequence of super-resolution models.

In particular, two notable contributions are the development of [noise conditioning augmentation](https://www.assemblyai.com/blog/how-imagen-actually-works/#robust-cascaded-diffusion-models), which noises low-resolution conditioning images in the super-resolution models, and [**dynamic thresholding**](https://www.assemblyai.com/blog/how-imagen-actually-works/#dynamic-thresholding) which helps prevent image saturation at high [classifier-free guidance](https://www.assemblyai.com/blog/how-imagen-actually-works/#classifier-free-guidance) weights.

![Dynamic Thresholding](images/dynamic_threshold.mp4)

See [How Imagen Actually Works](https://www.assemblyai.com/blog/how-imagen-actually-works/) for a detailed explanation of Imagen's operating principles.


## Attribution Note
This implementation is largely based on Phil Wang's [Imagen implementation](https://github.com/lucidrains/imagen-pytorch).
