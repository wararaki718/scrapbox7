from openai import OpenAI

from download import download_pdfs


def main() -> None:
    # setup data
    filepaths = download_pdfs()
    print(filepaths)
    print()

    # create vector store
    client = OpenAI()
    vector_store = client.beta.vector_stores.create(name="sample")
    print("## vector_store info:")
    print(f"id={vector_store.id}")
    print(f"name={vector_store.name}")
    print()

    # upload data
    streams = [open(filepath, "rb") for filepath in filepaths]
    file_batch = client.beta.vector_stores.file_batches.upload_and_poll(
        vector_store_id=vector_store.id, files=streams,
    )
    print("## vector_store file_batch:")
    print(f"id={file_batch.id}")
    print(f"status={file_batch.status}")
    print(f"vector_store_id={file_batch.vector_store_id}")
    print(f"n_files={file_batch.file_counts}")
    print()

    # create assistant & add vector_store resource
    assistant = client.beta.assistants.create(
        name="sample assistant",
        model="gpt-3.5-turbo",
        tools=[{"type": "file_search"}],
    )
    assistant = client.beta.assistants.update(
        assistant_id=assistant.id,
        tool_resources={"file_search": {"vector_store_ids": [vector_store.id]}}
    )

    # chat
    thread = client.beta.threads.create()
    prompt = "入院・通院をしている人の症状について教えてください。"
    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=prompt,
    )

    runner = client.beta.threads.runs.create_and_poll(
        thread_id=thread.id,
        assistant_id=assistant.id
    )
    
    messages = list(client.beta.threads.messages.list(thread_id=thread.id, run_id=runner.id))
    message_content = messages[0].content[0].text
    annotations = message_content.annotations
    citations = []
    for index, annotation in enumerate(annotations):
        message_content.value = message_content.value.replace(annotation.text, f"[{index}]")

        if citation := getattr(annotation, "file_citation", None):
            file = client.files.retrieve(citation.file_id)
            citations.append(f"[{index}] {file.filename}")
    
    print(message_content.value)
    print("\n".join(citations))

    print("DONE")


if __name__ == "__main__":
    main()
