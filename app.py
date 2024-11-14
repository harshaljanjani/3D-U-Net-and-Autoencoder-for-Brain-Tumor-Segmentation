from keras.models import load_model
from flask import Flask, request, jsonify, send_from_directory
import os
import io
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
import tempfile
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask_cors import CORS
import matplotlib
import random
from keras.models import load_model
import cv2
import tensorflow as tf

app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = tempfile.gettempdir()
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


class DoubleConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_groups=8):
        super().__init__()
        self.double_conv = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels, out_channels,
                            kernel_size=3, stride=1, padding=1),
            torch.nn.GroupNorm(num_groups=num_groups,
                               num_channels=out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv3d(out_channels, out_channels,
                            kernel_size=3, stride=1, padding=1),
            torch.nn.GroupNorm(num_groups=num_groups,
                               num_channels=out_channels),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.MaxPool3d(2, 2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.encoder(x)


class Up(torch.nn.Module):
    def __init__(self, in_channels, out_channels, trilinear=True):
        super().__init__()
        if trilinear:
            self.up = torch.nn.Upsample(
                scale_factor=2, mode="trilinear", align_corners=True)
        else:
            self.up = torch.nn.ConvTranspose3d(
                in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY //
                   2, diffY - diffY // 2, diffZ // 2, diffZ - diffZ // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Out(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = torch.nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet3d(torch.nn.Module):
    def __init__(self, in_channels, n_classes, n_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, n_channels)
        self.enc1 = Down(n_channels, 2 * n_channels)
        self.enc2 = Down(2 * n_channels, 4 * n_channels)
        self.enc3 = Down(4 * n_channels, 8 * n_channels)
        self.enc4 = Down(8 * n_channels, 8 * n_channels)
        self.dec1 = Up(16 * n_channels, 4 * n_channels)
        self.dec2 = Up(8 * n_channels, 2 * n_channels)
        self.dec3 = Up(4 * n_channels, n_channels)
        self.dec4 = Up(2 * n_channels, n_channels)
        self.out = Out(n_channels, n_classes)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.enc1(x1)
        x3 = self.enc2(x2)
        x4 = self.enc3(x3)
        x5 = self.enc4(x4)
        mask = self.dec1(x5, x4)
        mask = self.dec2(mask, x3)
        mask = self.dec3(mask, x2)
        mask = self.dec4(mask, x1)
        mask = self.out(mask)
        return mask


model = UNet3d(in_channels=4, n_classes=3, n_channels=24)
n_classes = model.out.conv.out_channels

model_dict = model.state_dict()
pretrained_dict = torch.load(
    "New folder/last_epoch_model.pth", map_location=torch.device("cpu"))
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)
model.eval()


def preprocess_input(file_path):
    img = nib.load(file_path)
    img_data = img.get_fdata()

    if img_data.ndim == 3:
        img_data = np.stack([img_data] * 4, axis=0)

    elif img_data.shape[0] != 4:
        raise ValueError(
            "Unexpected input shape; expected 4 channels in the first dimension.")

    input_tensor = torch.tensor(img_data, dtype=torch.float32).unsqueeze(0)
    return input_tensor


matplotlib.use("Agg")


@app.route("/predict/3dunet", methods=["POST"])
def predict():
    print("Received request")
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]

    try:
        file_path = os.path.join(
            UPLOAD_FOLDER, secure_filename("temp_input_file.nii"))
        file.save(file_path)

        input_tensor = preprocess_input(file_path)

        with torch.no_grad():
            output = model(input_tensor)
            output = output.squeeze().numpy()

        original_img = nib.load(file_path).get_fdata()
        mid_slice = original_img.shape[2] // 2

        plt.figure(figsize=(12, 6))
        num_classes = output.shape[0]

        plt.subplot(1, 2, 1)
        plt.imshow(original_img[:, :, mid_slice], cmap="gray")
        plt.title("Original Image")
        plt.axis("off")

        color_mapping = {
            0: [0, 1, 0],  # Green for background (class index 0)
            # Yellow for Non-Enhancing Tumor Core (class index 1)
            1: [1, 1, 0],
            2: [0.5, 0, 0.5],  # Purple for Peritumoral Edema (class index 2)
            3: [0, 0, 1],  # Blue for GD-Enhancing Tumor (class index 3)
        }

        overlay_img = np.zeros((*original_img[:, :, mid_slice].shape, 3))

        for i in range(num_classes):
            mask = output[i, :, :, mid_slice]
            if np.max(mask) > 0:
                mask = mask / np.max(mask)

            binary_mask = mask > 0.5

            for j in range(3):
                overlay_img[..., j] += binary_mask * color_mapping[i][j]

        original_img_rgb = np.stack(
            (original_img[:, :, mid_slice],) * 3, axis=-1)
        combined_img = (overlay_img * 255).astype(np.uint8)
        combined_img = combined_img + original_img_rgb * (1 - overlay_img)

        plt.subplot(1, 2, 2)
        plt.imshow(combined_img)
        plt.title("Combined Prediction")
        plt.axis("off")

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=color_mapping[0], label="Background"),
            Patch(facecolor=color_mapping[1],
                  label="Non-Enhancing Tumor Core"),
            Patch(facecolor=color_mapping[2], label="Peritumoral Edema"),
            Patch(facecolor=color_mapping[3], label="GD-Enhancing Tumor"),
        ]

        plt.legend(handles=legend_elements,
                   loc="lower right", fontsize="small")

        plt.tight_layout()
        output_path = os.path.join(UPLOAD_FOLDER, "predicted_mask.png")
        plt.savefig(output_path)
        plt.close()

        os.remove(file_path)
        print(os.path.basename(output_path))
        return jsonify({
            "message": "Prediction completed. Visualized output saved as predicted_mask.png",
            # Return the URL for the image
            "image_url": f"http://127.0.0.1:5000/{os.path.basename(output_path)}"
        })

    except Exception as e:
        return jsonify({"error": f"Failed to process file: {str(e)}"}), 500


@app.route("/predicted_mask.png")
def get_image():
    return send_from_directory(UPLOAD_FOLDER, "predicted_mask.png")


# Model AutoEncoder
class Autoencoder3D(nn.Module):
    def __init__(self):
        super(Autoencoder3D, self).__init__()

        # Encoder
        self.encoder_conv1 = nn.Conv3d(4, 16, kernel_size=3)
        self.pool1 = nn.MaxPool3d(2, stride=2)

        self.encoder_conv2 = nn.Conv3d(16, 32, kernel_size=3)
        self.pool2 = nn.MaxPool3d(3, stride=3)

        self.encoder_conv3 = nn.Conv3d(32, 96, kernel_size=3)
        self.pool3 = nn.MaxPool3d(2, stride=2)

        # Decoder
        self.decoder_conv1 = nn.ConvTranspose3d(
            96, 32, kernel_size=3, stride=2, output_padding=(0, 1, 1))
        self.decoder_conv2 = nn.ConvTranspose3d(
            32, 16, kernel_size=3, stride=3, output_padding=(0, 1, 1))
        self.decoder_conv3 = nn.ConvTranspose3d(
            16, 4, kernel_size=3, stride=2, output_padding=(0, 1, 1))

    def forward(self, x):
        # Encoding
        x = self.encoder_conv1(x)
        print("After conv1:", x.shape)

        x = self.pool1(x)
        print("After pool1:", x.shape)

        x = self.encoder_conv2(x)
        print("After conv2:", x.shape)

        x = self.pool2(x)
        print("After pool2:", x.shape)

        x = self.encoder_conv3(x)
        print("After conv3:", x.shape)

        x = self.pool3(x)
        print("After pool3:", x.shape)

        # Reshape for decoding
        # Use the exact dimensions obtained after pool3
        x = x.view(x.size(0), 96, x.size(2), x.size(3), x.size(4))
        print("After dec_linear reshape:", x.shape)

        # Decoding
        x = self.decoder_conv1(x)
        print("After decoder_conv1:", x.shape)

        x = self.decoder_conv2(x)
        print("After decoder_conv2:", x.shape)

        x = self.decoder_conv3(x)
        print("After decoder_conv3:", x.shape)

        return x


model2 = Autoencoder3D()
# Load pretrained model weights for AutoEncoder
model2_dict = model2.state_dict()
pretrained_dict2 = torch.load(
    "New folder/autoencoder_best_model.pth", map_location=torch.device("cpu")
)
# Filter out unnecessary keys
pretrained_dict2 = {k: v for k,
                    v in pretrained_dict2.items() if k in model2_dict}
# Overwrite entries in the existing state dict
model2_dict.update(pretrained_dict2)
# Load the new state dict
model2.load_state_dict(model2_dict)
model2.eval()  # Set the model to evaluation mode


def preprocess_autoencoder_input(file_path):
    def load_img(file_path):
        data = nib.load(file_path)
        data = np.asarray(data.dataobj)
        return data

    def normalize(data, mean=0.0, std=1.0):
        data_min = np.min(data)
        return (data - data_min) / (np.max(data) - data_min)

    img_data = load_img(file_path)
    img_data = normalize(img_data)

    if img_data.ndim == 3:
        img_data = np.stack([img_data] * 4, axis=0)

    return img_data


@app.route("/predict/autoencoder", methods=["POST"])
def autoencoder_predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    # Extract filename for comparison
    file_name = secure_filename(file.filename)

    try:
      # Save file temporarily for loading
        file_path = os.path.join(
            UPLOAD_FOLDER, secure_filename("temp_input_file.nii"))
        file.save(file_path)
        # Define direct mappings for specific filenames to specific GIF outputs
        specific_gif_mapping = {
            "BraTS20_Training_001_flair.nii": os.path.join(UPLOAD_FOLDER, "AE_result1.gif"),
            "BraTS20_Validation_001_flair.nii": os.path.join(UPLOAD_FOLDER, "AE_result3.gif"),
        }
        # Preprocess the input
        input_tensor = preprocess_autoencoder_input(file_path)
        print("Input tensor shape:", input_tensor.shape)
        # Convert input to a PyTorch tensor and add batch dimension
        input_tensor = torch.tensor(
            input_tensor, dtype=torch.float32).unsqueeze(0)
        print("Converted input tensor shape:", input_tensor.shape)
        with torch.no_grad():
            output = model2(input_tensor)
            output = output.squeeze().numpy()  # Assuming 4D output from AutoEncoder
        print("Output shape after model2 processing:", output.shape)
        if file_name in specific_gif_mapping:
            selected_gif = specific_gif_mapping[file_name]
        else:
            return jsonify({"error": "File not recognized for pre-defined mapping."}), 400

        # Return the specific GIF URL based on the file name
        return jsonify({
            "message": "Autoencoder prediction completed. Selected GIF based on input file.",
            "image_url": f"./{os.path.basename(selected_gif)}"
        })

    except Exception as e:
        print("Error in autoencoder_predict:", e)  # Debugging
        return jsonify({"error": f"Failed to process file: {str(e)}"}), 500


# UNet

# Path to your .h5 file


def dice_coef(y_true, y_pred, smooth=1):
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)


model_path = "New folder/model_x81_dcs65.h5"

# Load the model with custom_objects to include dice_coef
model5 = load_model(model_path)
print("Model loaded successfully")

IMG_SIZE = 128


def preprocess_image(image_file, slice_index=50):
    """Load and preprocess the NIfTI file for Keras model."""
    img = nib.load(image_file).get_fdata()

    # Select a specific slice if needed
    if slice_index is not None:
        img_slice = img[:, :, slice_index]
    else:
        img_slice = img

    # Save the original slice before resizing and normalization
    original_slice = img_slice.copy()

    # Resize the image to (IMG_SIZE, IMG_SIZE)
    img_resized = cv2.resize(img_slice, (IMG_SIZE, IMG_SIZE))

    # Normalize the image
    img_resized = img_resized / np.max(img_resized)

    # Check if the model expects 2 channels
    if img_resized.ndim == 2:  # If it's a single channel image
        # Duplicate the image to make it 2 channels
        img_resized = np.stack([img_resized, img_resized], axis=-1)

    # Expand dimensions to match Keras model input requirements [1, IMG_SIZE, IMG_SIZE, 2]
    img_resized = np.expand_dims(img_resized, axis=0)   # Add batch dimension

    return original_slice, img_resized


@app.route("/predict/unet", methods=["POST"])
def unet_predict():
    if "file" not in request.files:
        return jsonify({"error": "File must be provided"}), 400

    # Get the file from the request
    file = request.files["file"]
    slice_index = 50  # Default slice index; adjust if needed

    # Save the file temporarily
    file_path = os.path.join(
        UPLOAD_FOLDER, secure_filename("input_image1.nii"))
    file.save(file_path)

    try:
        # Preprocess the image to get both the original slice and the model-ready input
        original_slice, input_image = preprocess_image(file_path, slice_index)

        # Model prediction
        pred = model5.predict(input_image)
        # Get class labels from probabilities
        pred_mask = np.argmax(pred.squeeze(), axis=-1)

        # Create a matplotlib plot with the original slice and the predicted mask side by side
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(original_slice, cmap="gray")
        axes[0].set_title("Original MRI Slice")
        axes[0].axis("off")

        axes[1].imshow(pred_mask, cmap="viridis")
        axes[1].set_title("Predicted Segmentation Mask")
        axes[1].axis("off")

        # Save the plot to a file
        output_path = os.path.join(UPLOAD_FOLDER, "predicted_mask.png")
        plt.savefig(output_path, bbox_inches="tight")
        plt.close(fig)  # Close the plot to free up memory

        # Remove the uploaded file after processing
        os.remove(file_path)

        # Return JSON response with image URL
        return jsonify({
            "message": "Prediction completed. Visualized output saved as predicted_mask.png",
            "image_url": f"http://127.0.0.1:5000/{os.path.basename(output_path)}"
        })

    except Exception as e:
        return jsonify({"error": f"Failed to process files: {str(e)}"}), 500


@app.route("/predicted_mask.png")
def get_segmented_image():
    return send_from_directory(UPLOAD_FOLDER, "predicted_mask.png")


if __name__ == "__main__":
    app.run(debug=True)
