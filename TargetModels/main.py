"""Model registry and per-client architecture assignment.

Architectures live in one registry keyed by name. Constructor signatures differ --
some need to know the input size to compute a flatten dimension, ConvNet takes a
whole shape description -- so build_model() inspects each signature and supplies
only the arguments it accepts, rather than dispatching on a hardcoded list of which
names need which arguments.
"""
import inspect

from TargetModels.models import get_models
from TargetModels.MobileNetV2 import get_mobilenetv2
from TargetModels.MobileNetV3 import get_mobilenetv3
from TargetModels.ResNet import get_resnet_models
from TargetModels.ShuffleNetV2 import get_shufflenetv2
from TargetModels.SqueezeNet import get_squeezenet
from TargetModels.VGG import get_vgg_model

REGISTRY = {}
for _source in (get_models, get_mobilenetv2, get_mobilenetv3, get_shufflenetv2,
                get_squeezenet, get_resnet_models, get_vgg_model):
    REGISTRY.update(_source())

MODEL_NAMES = sorted(REGISTRY)

# ConvNet is configurable rather than fixed, so its shape has to be spelled out.
# These are the settings used by the dataset-distillation literature this follows.
CONVNET_SETTING = {
    'net_width':   128,
    'net_depth':   3,
    'net_act':     'relu',
    'net_norm':    'instancenorm',
    'net_pooling': 'avgpooling',
}

# Cycled over the clients when args.Model_Heterogeneity is set: client i gets
# HETEROGENEOUS[i % len(HETEROGENEOUS)]. Ordered smallest-to-largest is tempting
# but the assignment is positional, so changing the order changes which client
# holds which architecture -- and therefore which data it is paired with.
HETEROGENEOUS = ('AlexNet', 'vgg11', 'ConvNet')
HOMOGENEOUS = 'AlexNet'


def build_model(name, args):
    """Instantiate the architecture registered under `name`."""
    if name not in REGISTRY:
        raise ValueError(f"Unknown model {name!r}. Available: {', '.join(MODEL_NAMES)}")

    model = REGISTRY[name]
    accepted = inspect.signature(model).parameters
    kwargs = {'image_channels': args.image_channel, 'num_classes': args.num_classes}

    if name == 'ConvNet':
        kwargs.update(CONVNET_SETTING)
    # Two spellings of the same thing across the model families.
    for key in ('input_size', 'im_size'):
        if key in accepted:
            kwargs[key] = args.image_size

    return model(**kwargs)


def assign_architectures(args):
    """One architecture name per client."""
    models_names = list(HETEROGENEOUS) if args.Model_Heterogeneity else [HOMOGENEOUS]
    return [models_names[i % len(models_names)] for i in range(args.num_clients)]


def get_target_models(args):
    """Build every client's model. Returns (models, architecture names)."""
    names = assign_architectures(args)
    return [build_model(name, args) for name in names], names


# Kept as the name the rest of the code already imports.
get_model = build_model
