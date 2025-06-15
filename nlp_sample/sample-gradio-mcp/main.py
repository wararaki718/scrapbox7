import gradio

from letter import LetterCounter


def main() -> None:
    counter = LetterCounter()

    demo = gradio.Interface(
        fn=counter.count,
        inputs=[
            gradio.Textbox(label="Text", placeholder="Enter text here..."),
            gradio.Textbox(label="Letter", placeholder="Enter letter to count..."),
        ],
        outputs=gradio.Number(label="Count of Letter"),
        title="Letter Counter",
        description="A simple app to count occurrences of a letter in a given text.",
    )
    demo.launch(mcp_server=True)


if __name__ == "__main__":
    main()
