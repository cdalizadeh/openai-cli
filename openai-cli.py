#!/usr/bin/env python3

import argparse
import os
import openai
import readline

openai.api_key_path = './.env'

PROMPT = '> '

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

        self.model = 'gpt-3.5-turbo'
        if model:
            self.mode = model

    def speak(self, content):
        message = {'role': 'user', 'content': content}
        messages = self.messages + [message]

        response = openai.ChatCompletion.create(
            messages = messages,
            stream = self.stream,
            model = self.model,
        )

        if self.stream:
            collected_messages = []
            with ColorWriter(TextColor.WHITE):
                for chunk in response:
                    chunk_message = chunk['choices'][0]['delta']
                    collected_messages.append(chunk_message)
                    print(chunk_message.get('content', ''), end='')

            print()
            response_content = ''.join([m.get('content', '') for m in collected_messages])
            response_message = {'role': 'assistant', 'content': response_content}

        else:
            response_message = response['choices'][0]['message']
            response_content = response_message['content']
            finish_reason = response['choices'][0]['finish_reason']

            if finish_reason != 'stop':
                with ColorWriter(TextColor.YELLOW):
                    print(f'Unexpected finish reason: {finish_reason}')

            with ColorWriter(TextColor.CYAN):
                print(response_content)

        self.messages.append(message)
        self.messages.append(response_message)

    def __del__(self):
        print()

def main():
    parser = argparse.ArgumentParser('Start a conversation with an OpenAI language model')
    parser.add_argument('-4', '--gpt4', action='store_true', help='Use GPT-4')
    parser.add_argument('--no_stream', action='store_true', help='Do not use the OpenAI stream API')
    parser.add_argument('--proxy', help='Route requests to an intermediary proxy server')
    parser.add_argument('initial_query', nargs='*', help='Initial query for the model')
    args = parser.parse_args()

    model = ''
    if args.gpt4:
        model = 'gpt-4'

    stream = True
    if args.no_stream:
        stream=False

    if args.proxy:
        openai.api_base = args.proxy

    initial_query = ' '.join(args.initial_query)

    conversation = Conversation(model=model, stream=stream)

    try:
        if initial_query:
            with ColorWriter(TextColor.GREEN):
                print(PROMPT + initial_query)

            conversation.speak(initial_query)
            print()

        while True:
            with ColorWriter(TextColor.GREEN):
                query = input(PROMPT)

            if query == 'exit' or query == 'exit()':
                break
            elif query == '':
                continue
            elif query == 'reset' or query == 'reset()':
                conversation = Conversation(model=model)
                continue

            conversation.speak(query)
            print()

    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()
