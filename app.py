import os
import base64
import requests
import replicate
import gradio as gr
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
import json
import random

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")

if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY in environment.")
if not REPLICATE_API_TOKEN:
    raise ValueError("Missing REPLICATE_API_TOKEN in environment.")

# Global state
last_top = ""
last_bottom = ""
last_shoes = ""
last_jacket = ""
last_extras = []
last_gender = "unisex"
last_prompt = ""
last_top_img = None
last_bottom_img = None
last_shoes_img = None
last_jacket_img = None
suggestion_data = []

base_identity_prompt = (
    "A full-body editorial photo of a young Japanese woman in a Tokyo apartment, standing facing the camera naturally with relaxed posture. Neutral expression, realistic skin tone, long straight hair. iPhone 15 Pro photo with ISO 800, 85mm lens, f/1.8, shallow depth of field. Soft natural light from window, subtle film grain. White backdrop with soft shadow and texture. Full-body visible, including shoes. No cropping."
)

negative_prompt = (
    "cartoon, surreal, painterly, extra clothing, floating accessories, cropped feet, altered face, misshaped limbs, logos, abstract"
)

def describe_clothing_with_vision(image_path, item_type, show_logs=False):
    try:
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
        data_url = f"data:image/jpeg;base64,{image_data}"

        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }

        body = {
            "model": "gpt-4-turbo-2024-04-09",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this clothing item in 1 sentence with precision: fabric, color, cut, collar/sleeve/hem, logos, and fit. Don't guess use cases."},
                        {"type": "image_url", "image_url": {"url": data_url}}
                    ]
                }
            ],
            "max_tokens": 150
        }

        res = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=body)
        res.raise_for_status()
        desc = res.json()["choices"][0]["message"]["content"].strip()
        if show_logs:
            print(f"{item_type.capitalize()} description: {desc}")
        return desc, "female" if item_type in ["bottom", "jacket"] and "skirt" in desc.lower() else "unisex"
    except Exception as e:
        print(f"GPT-4o error for {item_type}: {e}")
        fallback = {
            "top": "a plain long-sleeve cotton shirt",
            "bottom": "a short black pleated skirt",
            "shoes": "a pair of white low-top sneakers",
            "jacket": "a simple light denim jacket"
        }
        return fallback[item_type], "unisex"

def generate_outfit_image(top_img, bottom_img, shoes_img, jacket_img=None, use_existing_seed=False, suggestion_appendix=""):
    global last_top, last_bottom, last_shoes, last_jacket, last_gender, last_prompt
    global last_top_img, last_bottom_img, last_shoes_img, last_jacket_img

    if not use_existing_seed:
        top_desc, _ = describe_clothing_with_vision(top_img, "top", show_logs=True)
        bottom_desc, _ = describe_clothing_with_vision(bottom_img, "bottom", show_logs=True)
        shoes_desc, _ = describe_clothing_with_vision(shoes_img, "shoes", show_logs=True)
        jacket_desc = ""
        if jacket_img:
            jacket_desc, _ = describe_clothing_with_vision(jacket_img, "jacket", show_logs=True)

        last_top, last_bottom, last_shoes, last_jacket = top_desc, bottom_desc, shoes_desc, jacket_desc
        last_top_img, last_bottom_img, last_shoes_img, last_jacket_img = top_img, bottom_img, shoes_img, jacket_img
        last_extras.clear()

        additions = ", ".join(filter(None, [jacket_desc]))
        additions_text = f" The outfit includes {additions}." if additions else ""

        last_prompt = f"{base_identity_prompt} She is wearing {top_desc}, {bottom_desc}, and {shoes_desc}.{additions_text}"

    full_prompt = f"{last_prompt} {suggestion_appendix}".strip()
    print("=== FULL PROMPT ===\n", full_prompt)

    output = replicate.run(
        "ideogram-ai/ideogram-v3-turbo",
        input={
            "prompt": full_prompt,
            "aspect_ratio": "2:3",
            "negative_prompt": negative_prompt,
            "steps": 60,
            "seed": random.randint(100000, 9999999)
        }
    )
    image_url = output[0] if isinstance(output, list) else output
    image_response = requests.get(image_url, timeout=30)
    if image_response.status_code == 200:
        return Image.open(BytesIO(image_response.content))
    return None

def generate_style_suggestions():
    global suggestion_data, last_top, last_bottom, last_shoes
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    prompt = (
        f"Top: {last_top}\nBottom: {last_bottom}\nShoes: {last_shoes}\n"
        "Suggest 3 realistic styling additions (e.g., add belt, earrings, jacket) to improve the look without changing the items."
    )

    body = {
        "model": "gpt-4-turbo-2024-04-09",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 300
    }
    try:
        res = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=body)
        res.raise_for_status()
        text = res.json()["choices"][0]["message"]["content"].strip()
        suggestion_data = [s.strip("0123456789. ") for s in text.split("\n") if s.strip()]
        return suggestion_data[:3]
    except Exception as e:
        print("Suggestion generation failed:", e)
        suggestion_data = []
        return []

def launch_app():
    with gr.Blocks() as demo:
        gr.Markdown("# 👗 AI Stylist App")
        with gr.Row():
            top = gr.Image(type="filepath", label="Top")
            bottom = gr.Image(type="filepath", label="Bottom")
            shoes = gr.Image(type="filepath", label="Shoes")
            jacket = gr.Image(type="filepath", label="Jacket (optional)")

        generate_btn = gr.Button("🎨 Generate Outfit Image")
        output_img = gr.Image(label="Generated Outfit", height=720)

        suggest_btn = gr.Button("🔄 Generate New Suggestions", visible=False)
        suggestion_rows = []
        suggestion_boxes = []
        try_buttons = []
        for i in range(3):
            with gr.Row(visible=False) as row:
                text = gr.Textbox(label=f"Suggestion {i+1}", interactive=False, scale=3)
                button = gr.Button(f"Try Suggestion {i+1}", scale=1)
                suggestion_rows.append(row)
                suggestion_boxes.append(text)
                try_buttons.append(button)

        apply_all_btn = gr.Button("✨ Try All Suggestions", visible=False)

        def run_generation(top, bottom, shoes, jacket):
            img = generate_outfit_image(top, bottom, shoes, jacket)
            updates = [gr.update(value="", visible=True) for _ in range(3)]
            return [img, gr.update(visible=True)] + updates + [gr.update(visible=True)] * 3 + [gr.update(visible=True)]

        def try_suggestion(index):
            return generate_outfit_image(
                top_img=last_top_img,
                bottom_img=last_bottom_img,
                shoes_img=last_shoes_img,
                jacket_img=last_jacket_img,
                use_existing_seed=True,
                suggestion_appendix=f" Additionally, she is accessorized with {suggestion_data[index]}."
            )

        def try_all():
            global last_extras
            last_extras = suggestion_data
            return generate_outfit_image(
                top_img=last_top_img,
                bottom_img=last_bottom_img,
                shoes_img=last_shoes_img,
                jacket_img=last_jacket_img,
                use_existing_seed=True,
                suggestion_appendix=" Additionally, she is accessorized with " + ", ".join(suggestion_data)
            )

        generate_btn.click(
            fn=run_generation,
            inputs=[top, bottom, shoes, jacket],
            outputs=[output_img, suggest_btn] + suggestion_boxes + suggestion_rows + [apply_all_btn]
        )

        for i, btn in enumerate(try_buttons):
            btn.click(fn=lambda idx=i: try_suggestion(idx), outputs=[output_img])

        suggest_btn.click(
            fn=lambda: [gr.update(value=s, visible=True) for s in generate_style_suggestions()],
            outputs=suggestion_boxes
        )

        apply_all_btn.click(fn=try_all, outputs=[output_img])

    demo.launch()

if __name__ == "__main__":
    launch_app()
