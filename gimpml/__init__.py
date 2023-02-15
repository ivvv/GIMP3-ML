from .complete_install import setup_python_weights
from .tools.coloring.coloring_idc import get_deepcolor as deepcolor
from .tools.deblur.deblur import get_deblur as deblur
from .tools.dehaze.dehaze import get_dehaze as dehaze
from .tools.denoise.denoise import get_denoise as denoise
from .tools.enlighten.enlighten import get_enlighten as enlighten
from .tools.edgedetect.edgedetect import get_edges as edge
from .tools.inpainting.inpainting import get_inpaint as inpaint
from .tools.interpolation.interpolation import get_inter as interpolateframe
from .tools.kmeans import get_kmeans as kmeans
from .tools.matting.matting import get_matting as matting
from .tools.semseg.monodepth import get_mono_depth as depth
from .tools.semseg.semseg import get_seg as semseg
from .tools.sresolution.superresolution import get_super as super
from .filters import *
__version__ = "0.0.9"
