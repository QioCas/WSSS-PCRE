from model.model_wsddn import network as network_wsddn
def build_network_baseline(args):
    from model.model_baseline import network_baseline

    model = network_baseline(args,
    backbone=args.backbone,
    num_classes=args.num_classes,
    pretrained=args.pretrained,
    aux_layer=args.aux_layer
    )
    param_groups = model.get_param_groups()

    return model, param_groups
def build_network(args):
    from model.model_seco import network

    model = network(args,
    backbone=args.backbone,
    num_classes=args.num_classes,
    pretrained=args.pretrained,
    init_momentum=args.momentum,
    aux_layer=args.aux_layer
    )
    param_groups = model.get_param_groups()

    return model, param_groups
def build_network_wsddn(args,get_seg=None):

    model = network_wsddn(args,
    backbone=args.backbone,
    num_classes=args.num_classes,
    pretrained=args.pretrained,
    init_momentum=args.momentum,
    aux_layer=args.aux_layer,
    get_seg=get_seg,
    )
    param_groups = model.get_param_groups()

    return model, param_groups