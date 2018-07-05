<h1>Artwork It. </h1><br>
A pytorch implementation of Artistic Neural Style Transfer.

This implementation is based on the paper <a href='https://arxiv.org/abs/1508.06576'>A Neural Algorithm of Artistic Style</a> by Leon A. Gatys,et. al. and <a href='https://alexis-jacq.github.io/'>Alexis Jacq's</a> wonderful <a href='https://pytorch.org/tutorials/advanced/neural_style_tutorial.html'>tutorial</a>. 

Check artist.py for the implementation. The code is self-explanatory with adequate comments.
artwork_it.py is supposed to be the CLI wrapper for the implementation which is yet to be coded. 

<h3> USAGE </h3>
<p><pre>
usage: artwork_it.py [-h] --content-image CONTENT_IMAGE --style-image
                     STYLE_IMAGE --out OUT [--img-init {0,1,2}]
                     [--style-layers STYLE_LAYERS]
                     [--content-layers CONTENT_LAYERS] [--iters ITERS]
                     [--verbose VERBOSE] [--log-every LOG_EVERY]
                     [--style-weight STYLE_WEIGHT]
                     [--content-weight CONTENT_WEIGHT]
                     [--cnn {vgg11,vgg13,vgg13_bn,vgg16,vgg16_bn,vgg19}]
                     [--im-size IM_SIZE]
</pre></p>
<h3>ARUGEMENTS</h3>
<p><pre>
	 -h, --help            show this help message and exit
  --content-image CONTENT_IMAGE, -ci CONTENT_IMAGE
                        Path to the content image. type: str
  --style-image STYLE_IMAGE, -si STYLE_IMAGE
                        path to style image. type: str
  --out OUT, -o OUT     output image path. type: str
  --img-init {0,1,2}, -init {0,1,2}
                        Image initialisation type. 0: Random Init, 1: Init
                        Content Image, 2: Init Style Image. Default=1 type:
                        int
  --style-layers STYLE_LAYERS, -sl STYLE_LAYERS
                        Layers where style loss is to be calculated. type:
                        list(string). e.g. conv_2 conv_3
  --content-layers CONTENT_LAYERS, -cl CONTENT_LAYERS
                        Layers where content loss is to be calculated. type:
                        list(string). e.g. conv_2 conv_3
  --iters ITERS, -iters ITERS
                        number of iterations the optimizer should be run.
                        Default=150 type: int
  --verbose VERBOSE, -v VERBOSE
                        To print stuff or not to print. Default=True type:
                        bool
  --log-every LOG_EVERY, -le LOG_EVERY
                        How often should stuff be printed. Default=50 type:
                        int
  --style-weight STYLE_WEIGHT, -sw STYLE_WEIGHT
                        Weight for the style image. Default=1e8 type: float
  --content-weight CONTENT_WEIGHT, -cw CONTENT_WEIGHT
                        Weight for the content image. Default=1e-1 type: float
  --cnn {vgg11,vgg13,vgg13_bn,vgg16,vgg16_bn,vgg19}, -nn {vgg11,vgg13,vgg13_bn,vgg16,vgg16_bn,vgg19}
                        What base CNN should be used. Default=vgg11 available:
                        vgg11, vgg13, vgg13_bn,vgg16,vgg16_bn,vgg19. type: str
  --im-size IM_SIZE, -im-size IM_SIZE
                        image size. default=(512,512) if cuda available, else
                        (300,300). type=tuple(int)
</pre></p>
<h3>DEPENDENCIES</h3>
<ul>
	<li>Python 3.x</li>
	<li>PyTorch</li>
	<li>numpy</li>
	<li>pyplot</li>
	<li>torchvision</li>
	<li>PIL</li>
</ul>
<h3>COMING SOON</h3>
<ul>
	<li>Better Documentation!</li>
	<li>A more modular implementation.</li>
	<li>A faster implementation and probably the ability to save and reuse a model.</li>
</ul>