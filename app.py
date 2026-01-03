
!pip install gradio groq

from google.colab import userdata 
import json
import gradio as gr
from groq import Groq

api_key = userdata.get("GROQ_API_KEY") 
client = Groq(api_key=api_key)


def explain_science_term(term: str):
    if not term.strip():
        return "Please enter a term.", "", ""

    system_instruction = """
You are a scientific assistant.
Your task is to explain complex concepts in simple terms.

You MUST return the response strictly in JSON format.
No additional text.

JSON structure:
{
  "definition": "Clear scientific definition",
  "analogy": "Simple analogy for beginners",
  "real_life_examples": "Examples of how this concept appears in everyday life"
}
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": f"Explain the term: {term}"}
            ],
            response_format={"type": "json_object"}
        )

        data = json.loads(response.choices[0].message.content)

        return (
            data.get("definition", ""),
            data.get("analogy", ""),
            data.get("real_life_examples", "")
        )

    except Exception as e:
        return f"Error: {e}", "Please try again.", "-"


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🧬 Science Explorer (AI)")
    gr.Markdown("Enter a scientific term to receive a clear explanation.")

    with gr.Row():
        with gr.Column(scale=1):
            input_term = gr.Textbox(
                label="Scientific Term",
                placeholder="Quantum mechanics"
            )
            explain_btn = gr.Button("Explain", variant="primary")

        with gr.Column(scale=2):
            output_def = gr.Textbox(label="Definition", lines=4)
            output_ana = gr.Textbox(label="Analogy", lines=4)
            output_examples = gr.Textbox(label="Real-life Examples", lines=6)

    explain_btn.click(
        fn=explain_science_term,
        inputs=input_term,
        outputs=[output_def, output_ana, output_examples]
    )


demo.launch()
