class Config:
    def __init__(self):
        # dataset
        self.load_data_path = '../data/'
        self.save_data_path = 'images'
        self.generator_feature_maps = 64
        self.discriminator_feature_maps = 64
        self.noise_dimension = 100
        self.image_size = 96

        self.env = 'GAN'

        # model
        self.load_generator = None
        self.load_discriminator = None

        # training
        self.batch_size = 64
        self.epoch = 40
        self.num_workers = 4
        self.lr1 = 0.0002
        self.lr2 = 0.0002
        self.beta1 = 0.5  # Adam优化器的beta1参数
        self.LAMBDA = 10  # 文献提供的超参数
        self.lr_decay = 0.95
        self.weight_decay = 0.99
        self.iter_count = 0
        self.epoch_count = 0

        self.generate_num = 64


def _parse(self, kwargs):  # **kwargs
    for key, value in kwargs.items():
        setattr(self, key, value)
    for key, value in self.__class__.__dict__.items():
        if not key.startswith("__"):
            print(key, value)


Config._parse = _parse
opt = Config()


