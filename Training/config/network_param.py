class NetworkParameter:
    CLASSES = ['scissor_center', 'gripper_center', 'scissor_hinge',
               'scissor_shaft_end', 'gripper_hinge', 'gripper_shaft_end']
    ENCODER = 'resnet18'  # 'resnet34'
    ENCODER_WEIGHTS = 'imagenet'
    ACTIVATION = 'sigmoid'
    DEVICE = 'cuda'

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
