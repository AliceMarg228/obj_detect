HIDDEN_UNITS = 45

class model_arch_v0(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.conv_layer = torch.nn.Sequential(

        torch.nn.Conv2d(
        in_channels = 3,
        out_channels = HIDDEN_UNITS,
        kernel_size = 3
    ),
        torch.nn.ReLU(),

        torch.nn.Conv2d(
        in_channels = HIDDEN_UNITS,
        out_channels = HIDDEN_UNITS,
        kernel_size = 3
    ),
        torch.nn.ReLU(),

        torch.nn.MaxPool2d(
            kernel_size = 2
        )

    )

    self.classifier = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(
            in_features= 64 * HIDDEN_UNITS,
            out_features = 4
        )
    )
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.classifier(self.conv_layer(x))

  def make_preds(self, x: torch.Tensor or nedo_image) -> torch.Tensor:
    """
      x = nedo_image instance or
      a tensor representing an image
      of a size (3, 20, 20)

      Returns:
        a tensor containing predicted
        values of x, y, w, h in that
        order
    """
    self.eval()
    with torch.inference_mode():
      if type(x) == nedo_image:
        return self.forward(x.img.unsqueeze(dim=0))
      if type(x) == torch.Tensor:
        return self.forward(x.unsqueeze(dim=0))
      else:
        raise TypeError("Expected torch.Tensor or nedo_image object, got ", type(x), " instead. ")