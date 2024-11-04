#!/usr/bin/env python3

import argparse
import os
import readline

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

PROMPT = '>> '
MULTILINE_PROMPT = ''

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
                if chunk_message != None:
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

def get_multi_input():
    with ColorWriter(TextColor.GREEN):
        print(PROMPT)

    lines = []
    while True:
        try:
            with ColorWriter(TextColor.GREEN):
                line = input(MULTILINE_PROMPT)
            lines.append(line)
        except EOFError:  # Ctrl-D pressed, input ends.
            print()
            break
    query = "\n".join(lines)
    return query

def main():
    parser = argparse.ArgumentParser('Start a conversation with an OpenAI language model')
    parser.add_argument('-3', '--gpt3', action='store_true', help='Use GPT-3.5')
    parser.add_argument('-m', '--multi', action='store_true', help='Start the conversation in multi mode')
    parser.add_argument('-t', '--terminate', action='store_true', help='Terminate the conversation after a single question')
    parser.add_argument('--proxy', help='Route requests to an intermediary proxy server')
    parser.add_argument('initial_query', nargs='*', help='Initial query for the model')
    args = parser.parse_args()

    model = 'gpt-3.5-turbo' if args.gpt3 else 'gpt-4o'
    multi_mode = args.multi
    terminate = args.terminate

    if args.proxy:
        openai.api_base = args.proxy

    initial_query = ' '.join(args.initial_query)

    conversation = Conversation(model=model, stream=True)

    try:
        while True:
            if initial_query:
                with ColorWriter(TextColor.GREEN):
                    print(PROMPT + initial_query)

                query = initial_query
                initial_query = False
                multi_mode = False

            elif multi_mode:
                query = get_multi_input()
                multi_mode = False

            else:
                with ColorWriter(TextColor.GREEN):
                    query = input(PROMPT)

            if query == '':
                continue

            elif query == 'exit' or query == 'exit()':
                break

            elif query in ('reset', 'reset()'):
                conversation = Conversation(model=model)
                continue

            elif query in ('multi', 'multi()', 'm'):
                query = get_multi_input()

            with ColorWriter(TextColor.WHITE):
                for msg in conversation.ask(query):
                    print(msg, end='')
                print()
                print()

            if terminate:
                break

    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()
