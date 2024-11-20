import os
import cv2
import gradio as gr
import numpy as np
import random
import base64
import requests
import json
import time
import spaces
from gradio_box_promptable_image import BoxPromptableImage
from gen_box_func import generate_parameters, visualize

import torch
from RAG_pipeline_flux import RAG_FluxPipeline

MAX_SEED = 999999

pipe = RAG_FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")

global run_nums

def update_run_num():
    with open("assets/run_num.txt", "r+") as f:
        run_num = int(f.read().strip()) + 1
        f.seek(0) 
        f.write(str(run_num))
    return run_num

# init
run_num = update_run_num()
def read_run_num():
    with open("assets/run_num.txt", "r+") as f:
        run_num = int(f.read().strip())
    return run_num

def get_box_inputs(prompts):

    # if isinstance(prompts, str):
    #     prompts = json.loads(prompts)
    if prompts=="layout1":
        prompts=[[0.05*1024, 0.05*1024, 2.0, (0.05+0.40)*1024, (0.05+0.9)*1024, 3.0], [0.5*1024, 0.05*1024, 2.0, (0.5+0.45)*1024, (0.05+0.9)*1024, 3.0]]
    elif prompts=="layout2":
        prompts=[[20.0, 425.0, 2.0, 551.0, 1008.0, 3.0], [615.0, 84.0, 2.0, 1000.0, 389.0, 3.0]]
    elif prompts=="layout3":
        prompts=[[0.2*1024, 0.1*1024, 2.0, (0.2+0.6)*1024, (0.1+0.4)*1024, 3.0],[0.2*1024, 0.6*1024, 2.0, (0.2+0.6)*1024, (0.6+0.35)*1024, 3.0]]
    elif prompts=="layout4":
        prompts=[[9.0, 153.0, 2.0, 343.0, 959.0, 3.0], [376.0, 145.0, 2.0, 692.0, 959.0, 3.0], [715.0, 143.0, 2.0, 1015.0, 956.0, 3.0]]
    box_inputs = []

    for prompt in prompts:
        # print("prompt",prompt)
        if prompt[2] == 2.0 and prompt[5] == 3.0:
            box_inputs.append((prompt[0], prompt[1], prompt[3], prompt[4]))
    return box_inputs

@spaces.GPU
def rag_gen(
    # box_prompt_image, 
    box_point,
    box_image,
    prompt, 
    coarse_prompt, 
    detailed_prompt, 
    HB_replace, 
    SR_delta, 
    num_inference_steps, 
    guidance_scale, 
    seed, 
    randomize_seed,
    ):
    points, image = box_point, box_image
    print("points", points)
    box_inputs = get_box_inputs(points)
    # prompt_img_height, prompt_img_width, _ = image.shape
    prompt_img_height, prompt_img_width = 1024,1024

    # GREEN = (36, 255, 12)

    HB_prompt_list = coarse_prompt.split("BREAK")


    HB_m_offset_list, HB_n_offset_list, HB_m_scale_list, HB_n_scale_list, SR_hw_split_ratio = generate_parameters(box_inputs, prompt_img_width, prompt_img_height)
    image = visualize(HB_m_offset_list, HB_n_offset_list, HB_m_scale_list, HB_n_scale_list, SR_hw_split_ratio, prompt_img_width, prompt_img_height)
    
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    else:
        seed = seed % MAX_SEED
    

    SR_prompt = detailed_prompt
    rag_image = pipe(
        SR_delta=SR_delta,
        SR_hw_split_ratio=SR_hw_split_ratio,
        SR_prompt=SR_prompt,
        HB_prompt_list=HB_prompt_list,
        HB_m_offset_list=HB_m_offset_list,
        HB_n_offset_list=HB_n_offset_list,
        HB_m_scale_list=HB_m_scale_list,
        HB_n_scale_list=HB_n_scale_list,
        HB_replace=HB_replace,
        seed=seed,
        prompt=prompt,
        height=1024,
        width=1024,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    ).images[0]
    global run_num
    run_num = update_run_num()

    # return image, rag_image, seed, f"<span style='font-size: 16px; font-weight: bold; color: red; display: block; text-align: center;'>Total inference runs: {run_num}</span>"
    # return rag_image, seed, f"<span style='font-size: 16px; font-weight: bold; color: red; display: block; text-align: center;'>Total inference runs: {run_num}</span>"
    return rag_image, seed

example_path = os.path.join(os.path.dirname(__file__), 'assets')

css="""
#col-left {
    margin: 0 auto;
    max-width: 400px;
}
#col-right {
    margin: 0 auto;
    max-width: 600px;
}
#col-showcase {
    margin: 0 auto;
    max-width: 1100px;
}
#button {
    color: blue;
}

#custom-label {
    color: purple; 
    font-size: 16px; 
    font-weight: bold; 
}
"""

assets_root_path = os.path.join(os.path.dirname(__file__), 'assets')

def load_description(fp):
    with open(fp, 'r', encoding='utf-8') as f:
        content = f.read()
    return content


with gr.Blocks(css=css) as demo:
    gr.HTML(load_description("assets/title.md"))
    
    # run_nums_box = gr.Markdown(
    #     value=f"<span style='font-size: 16px; font-weight: bold; color: red; display: block; text-align: center;'>Total inference runs: {run_num}</span>"
    # )
    
    with gr.Row():
        
        with gr.Column(elem_id="col-left"):
            gr.HTML("""
                <div style="display: flex; justify-content: center; align-items: center; text-align: center; font-size: 20px;">
                    <div>
                    
                    </div>
                    <div>
                    Step 1.  Choose
                     <span style="color: purple; font-weight: bold;">layout example</span>
                    </div>
                    
                </div>
            """)
                
            prompt = gr.Textbox(
                label="Prompt",
                placeholder="Enter your prompt",
                lines=2
            )

            coarse_prompt = gr.Textbox(
                label="Regional Fundamental Prompt(BREAK is a delimiter).",
                placeholder="Enter your prompt",
                lines=2
            )

            detailed_prompt = gr.Textbox(
                label="Regional Highly Descriptive Prompt(BREAK is a delimiter).",
                placeholder="Enter your prompt",
                lines=2
            )


        with gr.Column(elem_id="col-left"):
            
            default_image_path = "assets/images_template.png"
            box_image = gr.Image(
                show_label=False,
                interactive=False,
                label="Layout",
                value=default_image_path)
            # box_prompt_image = BoxPromptableImage(
            #     show_label=False,
            #     interactive=False,
            #     label="Layout",
            #     value={"image": default_image_path})
            # box_prompt_image = gr.Image(label="Layout", show_label=True)
            
            gr.HTML("""
            <div style="display: flex; justify-content: center; align-items: center; text-align: center; font-size: 16px;">
                <div style="display: flex; justify-content: center; align-items: center; text-align: center; font-size: 12px;">
                    <strong>
                        <span style="color: gray; font-weight: bold;">Tip: You can get a more ideal picture by adjusting HB_replace and SR_delta</span>
                    </strong>
                </div>
            </div>
            """)

        
            

        with gr.Column(elem_id="col-right"):
            
            gr.HTML("""
            <div style="display: flex; justify-content: center; align-items: center; text-align: center; font-size: 20px;">
                <div>
                Step 2. Press “Run” to get results 
                </div>
            </div>
            <div style="display: flex; justify-content: center; align-items: center; text-align: center; font-size: 10px;">
                <div>
                Errors may be displayed due to insufficient computing power
                </div>
            </div>
            """)
            
            # layout = gr.Image(label="Layout", show_label=True)

            result = gr.Image(label="Result", show_label=True)

            with gr.Accordion("Advanced Settings", open=False):         
                with gr.Row():
                    seed = gr.Slider(
                        label="Seed",
                        minimum=0,
                        maximum=MAX_SEED,
                        step=1,
                        value=0,
                    )
                    randomize_seed = gr.Checkbox(label="Random seed", value=True)
                
                with gr.Row():
                    HB_replace = gr.Slider(
                        label="HB_replace(The times of hard binding. More can make the position control more precise, but may lead to obvious boundaries.)",
                        minimum=0,
                        maximum=8,
                        step=1,
                        value=2,
                    )
                with gr.Row():
                    SR_delta = gr.Slider(
                        label="SR_delta(The fusion strength of image latent and regional-aware local latent. This is a flexible parameter, you can try 0.25, 0.5, 0.75, 1.0.)",
                        minimum=0.0,
                        maximum=1,
                        step=0.1,
                        value=1,
                    )
                
                with gr.Row():
                    guidance_scale = gr.Slider(
                        label="Guidance Scale",
                        minimum=1,
                        maximum=15,
                        step=0.1,
                        value=3.5,
                    )
    
                    num_inference_steps = gr.Slider(
                        label="Number of inference steps",
                        minimum=1,
                        maximum=50,
                        step=1,
                        value=20,
                    )

            with gr.Row():
                button = gr.Button("Run", elem_id="button")

    box_point = gr.Textbox(visible=False)
    gr.on(
        triggers=[
            button.click,
        ],
        fn=rag_gen,
        # fn=lambda *args: rag_gen(*args,[]),
        inputs=[
            box_point,
            box_image,
            prompt, 
            coarse_prompt, 
            detailed_prompt, 
            HB_replace, 
            SR_delta, 
            num_inference_steps, 
            guidance_scale, seed, 
            randomize_seed,
        ],
        # outputs=[layout, result, seed, run_nums_box],
        # outputs=[result, seed, run_nums_box],
        outputs=[result, seed],
        api_name="run",
    )

    with gr.Column():
        gr.HTML('<div id="custom-label">Layout Example ⬇️</div>')
        gr.Examples(
            # label="Layout Example (For more complex layouts, please run our code directly.)",
            examples=[
                [
                    "layout1",
                    "assets/case1.png",
                    "a man is holding a bag, a man is talking on a cell phone.",  # prompt
                    "A man holding a bag. BREAK a man holding a cell phone to his ear.",  # coarse_prompt
                    "A man holding a bag, gripping it firmly, with a casual yet purposeful stance. BREAK a man, engaged in conversation, holding a cell phone to his ear.",  # detailed_prompt
                    3,  # HB_replace
                    1.0,  # SR_delta
                    20,  # num_inference_steps
                    3.5,  # guidance_scale
                    1234,  # seed
                    False,  # randomize_seed
                ],
                [
                    "layout2",
                    "assets/case2.png", 
                    "A woman looking at the moon",  # prompt
                    "a woman BREAK a moon",  # coarse_prompt
                    "A woman, standing gracefully, her gaze fixed on the sky with a sense of wonder. BREAK The moon, luminous and full, casting a soft glow across the tranquil night.",  # detailed_prompt
                    3,  # HB_replace
                    0.8,  # SR_delta
                    20,  # num_inference_steps
                    3.5,  # guidance_scale
                    1233,  # seed
                    False,  # randomize_seed
                ],
                [
                    "layout3",
                    "assets/case3.png",
                    "a turtle on the bottom of a phone",  # prompt
                    "Phone BREAK Turtle",  # coarse_prompt
                    "The phone, placed above the turtle, potentially with its screen or back visible, its sleek design prominent. BREAK The turtle, below the phone, with its shell textured and detailed, eyes slightly protruding as it looks upward.",  # detailed_prompt
                    2,  # HB_replace
                    0.8,  # SR_delta
                    20,  # num_inference_steps
                    3.5,  # guidance_scale
                    1233,  # seed
                    False,  # randomize_seed
                ],
                [
                    "layout4",
                    "assets/case4.png",
                    "From left to right, a blonde ponytail Europe girl in white shirt, a brown curly hair African girl in blue shirt printed with a bird, an Asian young man with black short hair in suit are walking in the campus happily.",  # prompt
                    "A blonde ponytail European girl in a white shirt BREAK  A brown curly hair African girl in a blue shirt printed with a bird BREAK An Asian young man with black short hair in a suit",  # coarse_prompt
                    "A blonde ponytail European girl in a crisp white shirt, walking with a light smile. Her ponytail swings slightly as she enjoys the lively atmosphere of the campus. BREAK A brown curly hair African girl, her vibrant blue shirt adorned with a bird print. Her joyful expression matches her energetic stride as her curls bounce lightly in the breeze. BREAK An Asian young man in a sharp suit, his black short hair neatly styled, walking confidently alongside the two girls. His suit contrasts with the casual campus environment, adding an air of professionalism to the scene.",  # detailed_prompt
                    2,  # HB_replace
                    1.0,  # SR_delta
                    20,  # num_inference_steps
                    3.5,  # guidance_scale
                    1234,  # seed
                    False,  # randomize_seed
                ],
            ],
            inputs=[
                box_point,
                box_image,
                prompt, 
                coarse_prompt, 
                detailed_prompt, 
                HB_replace, 
                SR_delta, 
                num_inference_steps, 
                guidance_scale, 
                seed, 
                randomize_seed
            ],
            outputs=None,
            fn=rag_gen,
            cache_examples=False,
        )

if __name__ == "__main__":
    demo.queue(max_size=20).launch(share=True, server_port=7860)

