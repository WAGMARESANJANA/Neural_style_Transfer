import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image as kp_image

# Load and process images
def load_img(img_path, target_size=(224, 224)):
    img = Image.open(img_path).convert('RGB')
    img = img.resize(target_size, Image.LANCZOS)
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    return img

# Display images
def imshow(img, title=None):
    img = np.squeeze(img, axis=0)
    img = np.clip(img, 0.0, 1.0)
    plt.imshow(img)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()

# Preprocess content and style images
content_path = 'S:\CodeTech_Internship\content.jpg'
style_path = 'S:\CodeTech_Internship\style.jpg'

content_image = load_img(content_path)
style_image = load_img(style_path)

imshow(content_image, 'Content Image')
imshow(style_image, 'Style Image')

# Load VGG19 model
def get_model():
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False

    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
    content_layers = ['block5_conv2']
    selected_layers = style_layers + content_layers

    outputs = [vgg.get_layer(name).output for name in selected_layers]
    model = tf.keras.Model([vgg.input], outputs)
    
    return model, style_layers, content_layers

# Calculate Gram matrix
def gram_matrix(tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', tensor, tensor)
    return result / tf.cast(tensor.shape[1] * tensor.shape[2], tf.float32)

# Extract features
def get_feature_representations(model, content_image, style_image, style_layers, content_layers):
    # Process both images in a single batch
    style_outputs = model(style_image)
    content_outputs = model(content_image)

    # Compute Gram matrices for style
    style_features = [gram_matrix(output) for output in style_outputs[:len(style_layers)]]
    content_features = [output for output in content_outputs[len(style_layers):]]

    return style_features, content_features

# Compute total loss
def compute_loss(model, loss_weights, init_image, gram_style_features, content_features, style_layers, content_layers):
    style_weight, content_weight = loss_weights

    outputs = model(init_image)
    style_output_features = outputs[:len(style_layers)]
    content_output_features = outputs[len(style_layers):]

    style_score = 0
    content_score = 0

    # Style loss
    for target_gram, comb_output in zip(gram_style_features, style_output_features):
        gram_comb = gram_matrix(comb_output)
        style_score += tf.reduce_mean((gram_comb - target_gram) ** 2)

    # Content loss
    for target_content, comb_output in zip(content_features, content_output_features):
        content_score += tf.reduce_mean((comb_output - target_content) ** 2)

    style_score *= style_weight
    content_score *= content_weight

    total_loss = style_score + content_score
    return total_loss

# Training step
@tf.function()
def train_step(model, loss_weights, init_image, gram_style_features, content_features, style_layers, content_layers, optimizer):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, loss_weights, init_image, gram_style_features, content_features, style_layers, content_layers)
    
    grad = tape.gradient(loss, init_image)
    optimizer.apply_gradients([(grad, init_image)])
    return loss

# Setup and train
model, style_layers, content_layers = get_model()
style_features, content_features = get_feature_representations(model, content_image, style_image, style_layers, content_layers)

init_image = tf.Variable(content_image, dtype=tf.float32)
optimizer = tf.optimizers.Adam(learning_rate=0.02)
loss_weights = (1e-2, 1e4)  # (style_weight, content_weight)

# Style transfer loop
for i in range(1000):
    loss = train_step(model, loss_weights, init_image, style_features, content_features, style_layers, content_layers, optimizer)
    if i % 100 == 0:
        print(f'Step {i}, Loss: {loss.numpy():.4f}')
        imshow(init_image, f'Styled Image at Step {i}')

# Final image
imshow(init_image, 'Final Styled Image')