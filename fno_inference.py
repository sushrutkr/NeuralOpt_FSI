import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from omegaconf import OmegaConf
import hydra
import os
import vtk
from vtk.util.numpy_support import numpy_to_vtk
from modulus.models.fno import FNO  # Adjust import based on your model location
from hydra.utils import to_absolute_path
from modulus.launch.utils import load_checkpoint
from sklearn.preprocessing import MinMaxScaler

class FNOData(Dataset):
    def __init__(self, x):
        self.x = x
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, i):
        return self.x[i]

def load_model(checkpoint_path, cfg, device):
    model = FNO(
        in_channels=cfg.arch.fno.in_channels,
        out_channels=cfg.arch.decoder.out_features,
        decoder_layers=cfg.arch.decoder.layers,
        decoder_layer_size=cfg.arch.decoder.layer_size,
        dimension=cfg.arch.fno.dimension,
        latent_channels=cfg.arch.fno.latent_channels,
        num_fno_layers=cfg.arch.fno.fno_layers,
        num_fno_modes=cfg.arch.fno.fno_modes,
        padding=cfg.arch.fno.padding,
    ).to(device)
    
    # Load checkpoint using the modulus utility function
    load_checkpoint(path=checkpoint_path, device=device, models=model)
    model.eval()
    return model

def perform_inference(model, data_loader, device):
    predictions = []
    with torch.no_grad():
        for x_batch in data_loader:
            x_batch = x_batch.to(device)
            pred_batch = model(x_batch)
            predictions.append(pred_batch.cpu().numpy())
    predictions = np.concatenate(predictions, axis=0)
    return predictions

def save_vtk(predictions, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    # Iterate over timesteps
    for t in range(predictions.shape[0]):
        timestep_pred = predictions[t, 0]  # Selecting the prediction for timestep t
        dimensions = timestep_pred.shape  # Shape of the prediction
        
        # Create VTK grid and set points
        points = vtk.vtkPoints()
        values = vtk.vtkDoubleArray()
        values.SetName("PredictedField")

        for z in range(dimensions[2]):
            for y in range(dimensions[1]):
                for x in range(dimensions[0]):
                    points.InsertNextPoint(x, y, z)
                    values.InsertNextValue(timestep_pred[x, y, z])

        grid = vtk.vtkStructuredGrid()
        grid.SetDimensions(dimensions)
        grid.SetPoints(points)
        grid.GetPointData().SetScalars(values)

        # Write to VTK file
        writer = vtk.vtkStructuredGridWriter()
        vtk_filename = os.path.join(output_folder, f"predicted_field_timestep_{t}.vtk")
        writer.SetFileName(vtk_filename)
        writer.SetInputData(grid)
        writer.Write()

        print(f"Saved {vtk_filename}")

@hydra.main(config_path=".", config_name="config.yaml")
def main(cfg):
    input_file_path = to_absolute_path("data_fine.npy")  # Convert to absolute path
    checkpoint_path = to_absolute_path(cfg.scheduler.checkpoint_path)

    # Load Data
    try:
        scalar = MinMaxScaler(feature_range=(0, 1))
        data = np.load(input_file_path)
        shape = data.shape
        data = data.reshape(shape[0], -1)
        data = scalar.fit_transform(data)
        data = data.reshape(shape)

        x = data[:, 1:, :, :, :]  # Assuming this extracts the input data correctly
        print("x.shape:", x.shape)
    except FileNotFoundError:
        print(f"File {input_file_path} not found.")
        return

    # Initialize Dataset and DataLoader
    inference_dataset = FNOData(torch.tensor(x).float())
    inference_loader = DataLoader(inference_dataset, batch_size=cfg.training.batch_size, shuffle=False)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint_path, cfg, device)

    # Perform inference
    predictions = perform_inference(model, inference_loader, device)
    print("Predictions.shape:", predictions.shape)

    # Inverse transform predictions
    predictions = inverse_transform(predictions, scalar)

    # Save predictions as numpy array
    np.save("predictions.npy", predictions)

    # Save predictions as VTK files
    save_vtk(predictions, "vtk_results")

    print("Inference and saving complete.")

def inverse_transform(predictions, scaler):
    # predictions_new_shape = (15, 3, 77, 35, 48)
    predictions_new_shape = (15, 3, 153, 68, 98)
    
    # Create a new array with the desired shape filled with zeros (or any initial value)
    new_predictions = np.zeros(predictions_new_shape)

    # Copy data 3 times along the second axis (axis=1)
    new_predictions[:, 0, :, :, :] = predictions[:, 0, :, :, :]  # Copy the original data
    new_predictions[:, 1, :, :, :] = predictions[:, 0, :, :, :]  # Copy the original data again
    new_predictions[:, 2, :, :, :] = predictions[:, 0, :, :, :]  # Copy the original data again

    predictions = new_predictions
    # Reshape predictions to (15, -1) assuming 77x35x48 is the spatial dimension
    predictions = predictions.reshape(predictions.shape[0], -1)

    # Inverse transform predictions using the same scaler
    predictions_inv = scaler.inverse_transform(predictions)

    # Reshape back to the original shape (15, 1, 77, 35, 48)
    predictions_inv = predictions_inv.reshape(predictions.shape[0], 3, 153, 68, 98)
    predictions_inv = predictions_inv[:,:1,:,:,:]

    return predictions_inv




if __name__ == "__main__":
    main()
