import torch
from torchvision import transforms
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torchvision.models import efficientnet_v2_s, resnet18
from transformers import ViTFeatureExtractor, ViTForImageClassification
from flask import Flask, request, jsonify
from PIL import Image
import os

# Flask app setup
app = Flask(__name__)

# Load pre-trained models
# Image models
vit_feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
vit_model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
efficientnet_model = efficientnet_v2_s(pretrained=True)  # EfficientNet for skin tone
efficientnet_model.eval()

# Load pre-trained model for neckline classification (assuming it's trained on a fashion dataset)
# Fine-tuned ResNet-18 for neckline classification (V-Neck, Round Neck, etc.)
neckline_model = resnet18(pretrained=True)  # Placeholder for ResNet, should be fine-tuned
neckline_model.fc = torch.nn.Linear(neckline_model.fc.in_features, 5)  # 5 neckline classes
neckline_model.eval()

# Text model
roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
roberta_model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=3)  # 3 categories: Casual, Party, Bridal
roberta_model.eval()

# Transformations for EfficientNet, ResNet, and ViT
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Helper function: Predict skin tone using EfficientNet
@torch.no_grad()
def classify_image_efficientnet(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image_transform(image).unsqueeze(0)
    outputs = efficientnet_model(image)
    _, predicted_class = torch.max(outputs, 1)
    skin_tone_classes = ["Warm", "Cool", "Neutral"]
    return skin_tone_classes[predicted_class.item()]

# Helper function: Predict face shape using ViT
@torch.no_grad()
def classify_image_vit(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = vit_feature_extractor(images=image, return_tensors="pt")
    outputs = vit_model(**inputs)
    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    face_shape_classes = ["Oval", "Round", "Square", "Heart", "Rectangle"]
    return face_shape_classes[predicted_class]

# Helper function: Predict occasion using RoBERTa
@torch.no_grad()
def classify_text_occasion(text):
    inputs = roberta_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    outputs = roberta_model(**inputs)
    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    occasion_classes = ["Casual", "Party", "Bridal"]
    return occasion_classes[predicted_class]

# Helper function: Predict neckline using fine-tuned ResNet model
@torch.no_grad()
def classify_neckline(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image_transform(image).unsqueeze(0)
    outputs = neckline_model(image)
    _, predicted_class = torch.max(outputs, 1)
    neckline_classes = ["V-Neck", "Round Neck", "Square Neck", "Halter Neck", "Collared"]
    return neckline_classes[predicted_class.item()]

# Jewelry recommendation rules based on face shape, neckline, and occasion
def recommend_jewelry(face_shape, skin_tone, neckline, occasion):
    # Create a dictionary (hash map) for fast lookup
    # Original dataset
dataset = [
    {"face_shape": "Round", "skin_tone": "Warm", "neckline": "V-Neck", "occasion": "Bridal", "jewelry_type": "Kundan Necklace", "recommendation": "Layered Kundan necklaces with matching Kundan earrings for a traditional bridal look.", "jewelry_style": "Traditional"},
    {"face_shape": "Oval", "skin_tone": "Cool", "neckline": "Sweetheart", "occasion": "Party", "jewelry_type": "Polki Earrings", "recommendation": "Elegant Polki drop earrings to complement the rounded face.", "jewelry_style": "Traditional"},
    {"face_shape": "Heart", "skin_tone": "Neutral", "neckline": "Strapless", "occasion": "Casual", "jewelry_type": "Kundan Necklace", "recommendation": "A simple Kundan pendant necklace to avoid overwhelming the face.", "jewelry_style": "Traditional"},
    {"face_shape": "Square", "skin_tone": "Warm", "neckline": "High Neckline", "occasion": "Bridal", "jewelry_type": "Temple Jewelry Necklace", "recommendation": "Temple jewelry for a royal bridal look, with matching earrings and bangles.", "jewelry_style": "Traditional"},
    {"face_shape": "Oval", "skin_tone": "Cool", "neckline": "Sweetheart", "occasion": "Party", "jewelry_type": "Polki Earrings", "recommendation": "Dangling Polki earrings to create balance and elongate the face.", "jewelry_style": "Traditional"},
    {"face_shape": "Round", "skin_tone": "Neutral", "neckline": "V-Neck", "occasion": "Festive", "jewelry_type": "Kundan Necklace", "recommendation": "Traditional Kundan necklaces with colorful stones to add vibrancy to the look.", "jewelry_style": "Traditional"},
    {"face_shape": "Heart", "skin_tone": "Warm", "neckline": "Strapless", "occasion": "Casual", "jewelry_type": "Jhumkas", "recommendation": "Traditional Jhumkas (earrings) to add volume and beauty to the look.", "jewelry_style": "Traditional"},
    {"face_shape": "Neutral", "skin_tone": "Neutral", "neckline": "Scoop", "occasion": "Party", "jewelry_type": "Kundan Earrings", "recommendation": "Stunning Kundan earrings paired with a simple outfit for an elegant look.", "jewelry_style": "Traditional"},
    {"face_shape": "Round", "skin_tone": "Warm", "neckline": "Sweetheart", "occasion": "Bridal", "jewelry_type": "Necklace", "recommendation": "Long necklaces like Rani Haars to create a slimming effect.", "jewelry_style": "Modern"},
    {"face_shape": "Oval", "skin_tone": "Cool", "neckline": "Scoop", "occasion": "Casual", "jewelry_type": "Polki Pendant Necklace", "recommendation": "A delicate Polki pendant necklace to match a casual scoop neckline.", "jewelry_style": "Traditional"},
    {"face_shape": "Heart", "skin_tone": "Neutral", "neckline": "Strapless", "occasion": "Casual", "jewelry_type": "Necklace", "recommendation": "Avoid chokers; prefer delicate pendant necklaces.", "jewelry_style": "Modern"},
    {"face_shape": "Square", "skin_tone": "Warm", "neckline": "High Neckline", "occasion": "Bridal", "jewelry_type": "Necklace", "recommendation": "Choose soft rounded chokers or layered chains to soften angles.", "jewelry_style": "Modern"},
    {"face_shape": "Oval", "skin_tone": "Cool", "neckline": "Sweetheart", "occasion": "Party", "jewelry_type": "Earrings", "recommendation": "Dangling earrings to balance the face.", "jewelry_style": "Modern"},
    {"face_shape": "Round", "skin_tone": "Neutral", "neckline": "V-Neck", "occasion": "Festive", "jewelry_type": "Necklace", "recommendation": "Layered statement necklaces with gemstones to elongate the look.", "jewelry_style": "Modern"},
    {"face_shape": "Heart", "skin_tone": "Warm", "neckline": "Halter Neckline", "occasion": "Bridal", "jewelry_type": "Temple Jewelry Necklace", "recommendation": "A bold Temple necklace paired with matching Maang Tikka for a bridal look.", "jewelry_style": "Traditional"},
    {"face_shape": "Oval", "skin_tone": "Cool", "neckline": "Scoop", "occasion": "Casual", "jewelry_type": "Polki Pendant Necklace", "recommendation": "A delicate Polki pendant necklace to match a casual scoop neckline.", "jewelry_style": "Traditional"},
    {"face_shape": "Round", "skin_tone": "Warm", "neckline": "V-Neck", "occasion": "Bridal", "jewelry_type": "Necklace", "recommendation": "Long necklaces like Rani Haars to create a slimming effect.", "jewelry_style": "Modern"},
    {"face_shape": "Oval", "skin_tone": "Cool", "neckline": "Sweetheart", "occasion": "Casual", "jewelry_type": "Earrings", "recommendation": "Dangling earrings to balance the face; experiment with layered necklaces.", "jewelry_style": "Modern"},
    {"face_shape": "Heart", "skin_tone": "Neutral", "neckline": "Strapless", "occasion": "Casual", "jewelry_type": "Necklace", "recommendation": "Opt for delicate chains with small pendants.", "jewelry_style": "Modern"},
    {"face_shape": "Square", "skin_tone": "Warm", "neckline": "High Neckline", "occasion": "Bridal", "jewelry_type": "Necklace", "recommendation": "Choose soft rounded chokers or layered chains to soften angles.", "jewelry_style": "Modern"},
    {"face_shape": "Oval", "skin_tone": "Cool", "neckline": "Sweetheart", "occasion": "Party", "jewelry_type": "Earrings", "recommendation": "Dangling earrings to balance the face.", "jewelry_style": "Modern"},
    {"face_shape": "Round", "skin_tone": "Neutral", "neckline": "V-Neck", "occasion": "Festive", "jewelry_type": "Necklace", "recommendation": "Layered statement necklaces with gemstones to elongate the look.", "jewelry_style": "Modern"},
    {"face_shape": "Heart", "skin_tone": "Warm", "neckline": "Strapless", "occasion": "Casual", "jewelry_type": "Necklace", "recommendation": "Opt for delicate chains with small pendants.", "jewelry_style": "Modern"}
]

# Convert dataset into a jewelry_map dictionary
jewelry_map = {}

for item in dataset:
    key = (item['face_shape'].lower(), item['skin_tone'].lower(), item['neckline'].lower(), item['occasion'].lower())
    jewelry_map[key] = {
        "jewelry_type": item['jewelry_type'],
        "recommendation": item['recommendation'],
        "jewelry_style": item['jewelry_style']
    }

# Example to show the result
print(jewelry_map)
    
    # Look up the recommendation based on the input combination
    key = (face_shape.lower(), skin_tone.lower(), neckline.lower(), occasion.lower())
   
    # Check if the combination exists in the jewelry map
    if key in jewelry_map:
        return jewelry_map[key]["recommendation"], jewelry_map[key]["jewelry_style"]
    else:
        return "No recommendation found", ""

# API route for processing images and text
@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        # Handle image and text input from user
        image_file = request.files['image']
        occasion_text = request.form['text']

        # Save the uploaded image to a temporary file
        image_path = os.path.join('temp', image_file.filename)
        image_file.save(image_path)

        # Classify image and text using pre-trained models
        skin_tone = classify_image_efficientnet(image_path)
        face_shape = classify_image_vit(image_path)
        neckline = classify_neckline(image_path)
        occasion = classify_text_occasion(occasion_text)

        # Generate jewelry recommendation based on classifications
        jewelry_suggestion, jewelry_style = recommend_jewelry(face_shape, skin_tone, neckline, occasion)

        # Generate the final recommendation JSON
        recommendation = {
            "Face Shape": face_shape,
            "Skin Tone": skin_tone,
            "Neckline": neckline,
            "Occasion": occasion,
            "Jewelry Suggestion": jewelry_suggestion,
            "Jewelry Style": jewelry_style
        }

        return jsonify(recommendation)
   
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    # Ensure the temp directory exists
    os.makedirs('temp', exist_ok=True)
    app.run(debug=True)
