from torch import nn



def instanciarModeloMLP(config):

    modelo = RedMLP(
        full_l1=config["full_l1"],
        full_l2=config["full_l2"]
    )
    return  modelo


class RedMLP(nn.Module):

    def __init__(self, full_l1, full_l2) -> None:
        super().__init__()
        self.cantidad_entradas = 3 * 32 * 32
        self.denseLayer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=self.cantidad_entradas, out_features=full_l1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(in_features=full_l1, out_features=full_l2),
            nn.ReLU(),
            nn.Linear(in_features=full_l2, out_features=10)
        )

    def forward(self,x):
        logits = self.denseLayer(x)
        return logits