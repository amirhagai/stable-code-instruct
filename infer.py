import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gradio as gr

tokenizer = AutoTokenizer.from_pretrained("stabilityai/stable-code-instruct-3b", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("stabilityai/stable-code-instruct-3b", torch_dtype=torch.bfloat16, trust_remote_code=True)
model.eval()
model = model.cuda()

messages = [
    {
        "role": "system",
        "content": "You are a helpful and polite assistant",
    },
    {
        "role": "user",
        "content": "Write python code that takes center point, width and height and generate bbox"
    },
]

def update_user_content(new_content):
    for message in messages:
        if message["role"] == "user":
            message["content"] = new_content
            break  # Exit the loop after updating



def infer(user_content):

    update_user_content(user_content)
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

    tokens = model.generate(
        **inputs,
        max_new_tokens=1024,
        temperature=0.5,
        top_p=0.95,
        top_k=100,
        do_sample=True,
        use_cache=True
    )

    output = tokenizer.batch_decode(tokens[:, inputs.input_ids.shape[-1]:], skip_special_tokens=False)[0]
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return output

if __name__ == "__main__":

    demo = gr.Interface(
        fn=infer,
        inputs=[gr.Textbox(label="question", lines=40)],
        outputs=[gr.Textbox(label="code and usage", lines=60)],
    )
    demo.launch(share=True)


    #demo = gr.Interface(
     #   fn=infer,
      #  inputs=["text"],
       # outputs=["text"],
    #)

    #demo.launch(share=True)

    #new_user_content = ""
    #while new_user_content !="exit":
    #    new_user_content = input("Enter new user content: ")
    #    update_user_content(new_user_content)
    #    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    #    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

     #   tokens = model.generate(
     #       **inputs,
     #       max_new_tokens=1024,
     #       temperature=0.5,
     #       top_p=0.95,
     #       top_k=100,
     #       do_sample=True,
     #       use_cache=True
     #   )

      #  output = tokenizer.batch_decode(tokens[:, inputs.input_ids.shape[-1]:], skip_special_tokens=False)[0]
      #  print(output)
      #  torch.cuda.synchronize()
