#!/usr/bin/env python3

import argparse
import os
import tempfile
import subprocess

from dotenv import load_dotenv
from openai import OpenAI
from prompt_toolkit import PromptSession
from prompt_toolkit.enums import EditingMode
from prompt_toolkit.styles import Style
from prompt_toolkit.key_binding import KeyBindings

load_dotenv()

# Define color constants
class TextColor:
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    RESET = '\033[0m'

    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    ITALIC = '\033[3m'

PROMPT = '>> '
PROMPT_COLOR_ANSI = TextColor.GREEN
PROMPT_COLOR_PT = 'ansigreen'

class ColorWriter:
    def __init__(self, color):
        self.color = color

    def __enter__(self):
        print(self.color, end='')

    def __exit__(self, *args):
        print(TextColor.RESET, end='')

class Conversation:
    def __init__(self, stream=True, model=''):
        self.messages = []
        self.stream = stream
        self.model = 'gpt-4o' if not model else model
        self.client = OpenAI()

    def ask(self, content):
        message = {'role': 'user', 'content': content}
        messages = self.messages + [message]

        response = self.client.chat.completions.create(
            messages = messages,
            stream = self.stream,
            model = self.model,
        )

        if self.stream:
            collected_messages = []

            for chunk in response:
                chunk_message = chunk.choices[0].delta.content
                if chunk_message is not None:
                    collected_messages.append(chunk_message)
                    yield chunk_message

            response_content = ''.join(collected_messages)
            response_message = {'role': 'assistant', 'content': response_content}
            self.messages.extend([message, response_message])
        else:
            response_message = response.choices[0].message
            response_content = response_message.content
            finish_reason = response.choices[0].finish_reason

            if finish_reason != 'stop':
                raise Exception(f'Unexpected finish reason: {finish_reason}')

            yield response_content
            self.messages.extend([message, response_message])

def open_editor_with_content(initial_content=''):
    """Open the default editor with initial content and return the edited content."""
    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix='.md', delete=False) as tmp:
        tmp_path = tmp.name
        # Write the initial content to the file
        tmp.write(initial_content.encode('utf-8'))

    try:
        # Get the default editor from environment or use a fallback
        editor = os.environ.get('EDITOR', 'vim')

        # Open the editor with the temporary file
        subprocess.run([editor, tmp_path], check=True)

        # Read the content after editing
        with open(tmp_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Remove trailing newlines to avoid blank lines
        content = content.rstrip('\n')

        return content
    finally:
        # Clean up the temporary file
        os.unlink(tmp_path)

def get_input():
    # Define style for the prompt and input using the shared constant
    style = Style.from_dict({
        'prompt': PROMPT_COLOR_PT,  # Green color for prompt from shared constant
        '': PROMPT_COLOR_PT,        # Green color for user input from shared constant
    })

    # Create key bindings for Ctrl-T
    kb = KeyBindings()

    @kb.add('c-t')
    def _(event):
        """Open editor when Control-T is pressed."""
        # Get the current buffer text to pass to the editor
        buffer_text = event.app.current_buffer.text

        # Open the editor with the current buffer content
        edited_text = open_editor_with_content(buffer_text)

        # Put the edited text directly into the buffer
        event.app.current_buffer.document = event.app.current_buffer.document.__class__(
            edited_text,
            cursor_position=len(edited_text)
        )

        # Exit with the edited text as result
        event.app.exit(result=edited_text)

    # Configure prompt_toolkit session with vi editing mode
    session = PromptSession(
        editing_mode=EditingMode.VI,  # VI editing mode
        key_bindings=kb  # Add our key bindings
    )

    # Use prompt_toolkit with styled prompt and input
    query = session.prompt(
        [('class:prompt', PROMPT)],
        style=style,
        complete_while_typing=False,
        enable_history_search=True
    )

    return query

def main():
    parser = argparse.ArgumentParser('Start a conversation with an OpenAI language model')
    parser.add_argument('-3', '--gpt3', action='store_true', help='Use GPT-3.5')
    parser.add_argument('-t', '--terminate', action='store_true', help='Terminate the conversation after a single question')
    parser.add_argument('--proxy', help='Route requests to an intermediary proxy server')
    parser.add_argument('initial_query', nargs='*', help='Initial query for the model')
    args = parser.parse_args()

    model = 'gpt-3.5-turbo' if args.gpt3 else 'gpt-4o'
    terminate = args.terminate

    if args.proxy:
        os.environ['OPENAI_API_BASE'] = args.proxy

    initial_query = ' '.join(args.initial_query)

    conversation = Conversation(model=model, stream=True)

    try:
        while True:
            if initial_query:
                # For consistency with prompt_toolkit styling, keep entire line green
                print(f"{PROMPT_COLOR_ANSI}{PROMPT}{initial_query}{TextColor.RESET}")
                query = initial_query
                initial_query = None
            else:
                query = get_input()

            if query == '':
                continue

            elif query in ('exit', 'exit()'):
                break

            elif query in ('reset', 'reset()'):
                conversation = Conversation(model=model)
                continue

            color = TextColor.WHITE
            newline = True
            for msg in conversation.ask(query):
                if newline and msg.startswith('#'):
                    color = TextColor.MAGENTA

                if msg.endswith('\n'):
                    newline = True
                    color = TextColor.WHITE
                else:
                    newline = False

                with ColorWriter(color):
                    print(msg, end='')
            print()
            print()

            if terminate:
                break

    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()
