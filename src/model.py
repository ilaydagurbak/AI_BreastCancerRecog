from torch import nn


class Model(nn.Module):
    def __init__(self, original_model, num_classes):
        super(Model, self).__init__()
        # self.features = original_model.features

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(256 * 3* 3, 4096),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p = 0.2),
            nn.Linear(4096, 4096),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p = 0.2),
            nn.Linear(4096, 4096),
            nn.LeakyReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )



        self.modelName = 'alexnet'

        # for p in self.features.parameters() :
        #     p.requires_grad = False


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 *3  * 3)
        x = self.classifier(x)
        return x
