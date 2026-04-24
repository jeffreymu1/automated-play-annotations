import marimo

__generated_with = "0.23.2"
app = marimo.App(width="medium")


@app.cell
def _():
    from deepsport_dataset import DeepSportDataset
    import timm

    return DeepSportDataset, timm


@app.cell
def _(DeepSportDataset, timm):
    dataset = DeepSportDataset('data/deepsport-dataset')

    model = timm.create_model('convnext_base.dinov3_lvd1689m', pretrained=True, features_only=True)
    model = model.eval()
    return dataset, model


@app.cell
def _(dataset):
    X, y, calib = dataset[0]
    X = X[:, :, :, :].permute(0, 3, 1, 2).mean(axis=1)
    return (X,)


@app.cell
def _(X, model, timm):
    import torch

    data_config = timm.data.resolve_model_data_config(model)
    mean = torch.tensor(data_config["mean"]).view(1, 3, 1, 1)
    std = torch.tensor(data_config["std"]).view(1, 3, 1, 1)

    X_norm = (X.float() / 255.0 - mean) / std
    with torch.inference_mode():
        features = model(X_norm)
    return (features,)


@app.cell
def _(features):
    features[3].shape
    return


if __name__ == "__main__":
    app.run()
