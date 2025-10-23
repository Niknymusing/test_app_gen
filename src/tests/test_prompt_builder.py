import yaml
import os
from src.resources.prompt_builder.base import PromptBuilder
from src.resources.prompt_builder.retrievers import FileContextRetriever

def test_prompt_builder_with_context():
    # 1. Create a dummy context file
    context_content = "This is the context from the file."
    context_file_path = "test_context.txt"
    with open(context_file_path, "w") as f:
        f.write(context_content)

    # 2. Instantiate retrievers and builder
    file_retriever = FileContextRetriever(context_file_path)
    prompt_builder = PromptBuilder(context_retrievers=[file_retriever])

    # 3. Define a sample YAML prompt
    prompt_yaml_str = """
messages:
  - role: "system"
    template: "You are a helpful assistant. Here is some context: {context}"
  - role: "human"
    template: "My question is: {query}"
"""
    prompt_yaml = yaml.safe_load(prompt_yaml_str)

    # 4. Build the prompt
    query = "What is the capital of France?"
    messages = prompt_builder.build_from_yaml(
        prompt_yaml,
        format_args={"query": query},
        query_for_context=query
    )

    # 5. Assertions
    assert len(messages) == 2
    system_message = messages[0]
    human_message = messages[1]

    expected_system_content = f"You are a helpful assistant. Here is some context: {context_file_path}:\\n{context_content}"

    # Handle both dict and object messages
    if isinstance(system_message, dict):
        assert expected_system_content in system_message['content']
        assert query in human_message['content']
    else:
        assert expected_system_content in system_message.content
        assert query in human_message.content


    # Clean up the dummy file
    os.remove(context_file_path)
