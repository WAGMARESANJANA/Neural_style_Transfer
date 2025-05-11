# 🖼️ Neural Style Transfer with TensorFlow

This repository implements **Neural Style Transfer (NST)** using TensorFlow and a pre-trained VGG19 network. The goal is to take two images—a **content image** and a **style image**—and generate a new image that combines the content of the first with the artistic style of the second.

---

## 📌 Features

* Uses a pre-trained VGG19 network for feature extraction
* Calculates **content** and **style loss** independently
* Supports Gram matrix computation for capturing style features
* Trains via gradient descent using Adam optimizer
* Periodic output display to track progress
* Final image output showcasing the applied style

---

## 📂 Project Structure

```
.
├── neural_style_transfer.py  # Main script
├── content.jpg               # Input content image (provide your own)
├── style.jpg                 # Input style image (provide your own)
└── README.md                 # This file
```

---

## 🧰 Requirements

* Python 3.7+
* TensorFlow 2.x
* NumPy
* PIL (Pillow)
* Matplotlib

Install the required packages:

```bash
pip install tensorflow numpy pillow matplotlib
```

---

## 🚀 Getting Started

1. **Clone the repository:**

```bash
git clone https://github.com/yourusername/neural-style-transfer.git
cd neural-style-transfer
```

2. **Replace the input images:**

Place your `content.jpg` and `style.jpg` in the same directory as the script. Update the paths if needed.

3. **Run the script:**

```bash
python neural_style_transfer.py
```

During training, intermediate images will be displayed every 100 steps, and the final styled image will be shown at the end.

---

## 🧠 How It Works

1. **Image Loading and Preprocessing:**
   The content and style images are loaded, resized, and normalized to be compatible with VGG19.

2. **Model Setup:**
   VGG19 (without top layers) is used to extract features from specific layers that represent content and style.

3. **Feature Extraction:**

   * Content features are extracted from a deeper layer (`block5_conv2`).
   * Style features are extracted from multiple shallower layers and converted into **Gram matrices**.

4. **Loss Calculation:**

   * **Style Loss:** Measures the distance between Gram matrices of the generated and style images.
   * **Content Loss:** Measures the distance between content features of the generated and content images.
   * Total loss is a weighted combination of style and content losses.

5. **Optimization:**
   The image is treated as a trainable variable and updated using gradient descent to minimize the loss.

---

## ⚙️ Configuration

You can modify the following parameters inside the script:

* `target_size=(224, 224)`: Resize dimension of input images
* `style_layers`: Layers used to compute style loss
* `content_layers`: Layers used to compute content loss
* `loss_weights = (1e-2, 1e4)`: Tuple of style and content weights
* `learning_rate`: Learning rate for the Adam optimizer
* `range(1000)`: Number of optimization steps

---

## 📝 Notes

* Ensure the input images are not too large to avoid memory issues.
* Intermediate visualizations help monitor style transfer progress.
* Final image quality depends on tuning of style/content weights and number of steps.

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
