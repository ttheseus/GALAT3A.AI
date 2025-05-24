from transformers import Blip2Processor, Blip2ForConditionalGeneration, CLIPProcessor, CLIPModel
from PIL import Image
import torch
import subprocess

# load BLIP model and processor for captioning
blip2_processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-base")
blip2_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-base")

# load CLIP model and processor for semantic similarity
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

# load image
image_path = "C:/Users/Fanzh/OneDrive/Desktop/Code projects/AI/CritiqueAI/IMG_6855.png"
image = Image.open(image_path).convert("RGB")

# generate BLIP caption
inputs = processor(images=image, text="Describe this image.", return_tensors="pt")
outputs = model.generate(**inputs)
blip2_caption = processor.decode(outputs[0], skip_special_tokens=True)

print("📷 BLIP-2 Caption:", blip2_caption)

# define semantic descriptions for CLIP scoring
text_descriptions = [
    "a well-composed painting",
    "a cluttered image",
    "a beautiful color scheme",
    "a dull color scheme",
    "a focused composition",
    "a chaotic composition",
    "a portrait"
]

# use CLIP to score semantic descriptions against the image
inputs_clip = clip_processor(text=text_descriptions, images=image, return_tensors="pt", padding=True)
outputs = clip_model(**inputs_clip)
probs = outputs.logits_per_image.softmax(dim=1)[0].tolist()

# get top 3 CLIP descriptions
ranked = sorted(zip(text_descriptions, probs), key=lambda x: -x[1])
print("\n🔍 CLIP Semantic Analysis:")
for desc, prob in ranked[:3]:
    print(f"- {desc}: {prob:.2%}")

# format for LLM
prompt = (
    "You are an art critique AI. Given the image has these characteristics: "
    + ", ".join([desc for desc, _ in ranked[:3]])
    + ", give constructive critique focused on composition, clarity, and color use. Furthermore, point out specific areas of the artwork that could be improved without generalizing to 'backgroun' or 'subject'. Specifically state where needed improvement."
)

# run Mistral via Ollama
try:
    result = subprocess.run(
        ["ollama", "run", "mistral"],
        input=prompt.encode(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=100
    )
    print("\n🧠 LLM Critique (Mistral via Ollama):")
    print(result.stdout.decode(errors="ignore"))
except Exception as e:
    print(f"Error running LLM: {e}")
