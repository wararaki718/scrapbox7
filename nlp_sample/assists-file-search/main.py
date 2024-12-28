from openai import OpenAI
from openai.types.beta import Thread

from download import download_pdf
from handler import EventHandler


def main() -> None:
    # create client
    client = OpenAI()
    assistant = client.beta.assistants.create(
        name="Financial Analyst Assistant",
        model="gpt-3.5-turbo",
        tools=[{"type": "file_search"}],
    )

    # Create a vector store caled "Financial Statements"
    vector_store = client.beta.vector_stores.create(name="Financial Statements")

    # upload to OpenAI
    download_path = download_pdf()
    file_streams = [open(download_path, "rb"), open(download_path, "rb")]

    file_batch = client.beta.vector_stores.file_batches.upload_and_poll(
        vector_store_id=vector_store.id, files=file_streams
    )
    print(f"file status: {file_batch.status}")
    print(f"file counts: {file_batch.file_counts}")

    # update assitant
    assistant = client.beta.assistants.update(
        assistant_id=assistant.id,
        tool_resources={"file_search": {"vector_store_ids": [vector_store.id]}},
    )

    # Upload the user provided file to OpenAI
    message_file = client.files.create(file=open(download_path, "rb"), purpose="assistants")
    messages = [{
        "role": "user",
        "content": "How many shares of AAPL were outstanding at the end of of October 2023?",
        # Attach the new file to the message.
        "attachments": [{"file_id": message_file.id, "tools": [{"type": "file_search"}]}],
    }]

    # Create a thread and attach the file to the message
    thread: Thread = client.beta.threads.create(messages=messages)
    print(thread.tool_resources.file_search)

    with client.beta.threads.runs.stream(
        thread_id=thread.id,
        assistant_id=assistant.id,
        instructions="Please adddress the user as Jane Doe. The user has a premium account.",
        event_handler=EventHandler(),
    ) as stream:
        stream.until_done()

    print("DONE")


if __name__ == "__main__":
    main()
