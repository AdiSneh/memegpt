import json
import webbrowser
from enum import Enum
from pathlib import Path
from typing import Any, List, Dict

import openai
import requests
from pydantic import BaseModel, BaseSettings


class Settings(BaseSettings):
    openai_api_key: str
    temperature: int = 1
    model: str = "gpt-3.5-turbo"
    imgflip_username: str
    imgflip_password: str


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Message(BaseModel):
    role: Role
    content: str


class Template(BaseModel):
    id: int
    name: str
    captions: List[str]


class MemeRequest(BaseModel):
    scenario: str
    template: Template


settings = Settings()

openai.api_key = settings.openai_api_key
TEMPLATES_PATH = Path(__file__).parent / 'resources' / 'templates.json'
TEMPLATES = [
    Template.parse_obj(raw_template)
    for raw_template in json.loads(TEMPLATES_PATH.read_text())
]

NAME_TO_TEMPLATE = {template.name: template for template in TEMPLATES}
JOINED_TEMPLATE_NAMES = ", ".join(f"'{template.name}'" for template in TEMPLATES)

BASE_TEMPLATE_MESSAGES = [
    Message(
        role=Role.SYSTEM,
        content=(
            "You are an assistant designed to create memes. "
            "Given a scenario, you will respond with the name of a well known meme "
            "template that can be used to create a good and funny meme for that scenario. "
            f"You must choose one of the following templates: {JOINED_TEMPLATE_NAMES}"
        ),
    ),
    Message(
        role=Role.USER,
        content=(
            "My parents are always mad at me for playing video games because they think they will make me violent. "
            "But all I ever play is Minecraft and I'm just building houses and stuff..."
        ),
    ),
    Message(role=Role.ASSISTANT, content="Woman Yelling At Cat"),
    Message(
        role=Role.USER,
        content=(
            "All those dumb employers are not willing to pay an extra 20K to give their good employees a raise. "
            "So then the good workers quit and the employers end up paying 50K extra to hire new workers."
        ),
    ),
    Message(role=Role.ASSISTANT, content="Drake Hotline Bling"),
]

BASE_CAPTIONS_MESSAGES = [
    Message(
        role=Role.SYSTEM,
        content=(
            "You are an assistant designed to create memes. "
            "You will be given a scenario, a well known meme template and the names of the captions for that template. "
            "You will respond with the text for each caption in order to create "
            "a good and funny meme for the given scenario. "
            "You communicate in JSON format."
        ),
    ),
    Message(
        role=Role.USER,
        content=(
            MemeRequest(
                template=NAME_TO_TEMPLATE["Woman Yelling At Cat"],
                scenario=(
                    "My parents are always mad at me for playing video games "
                    "because they think they will make me violent. "
                    "But all I ever play is Minecraft and I'm just building houses and stuff..."
                ),
            ).json()
        ),
    ),
    Message(
        role=Role.ASSISTANT,
        content=json.dumps(
            {
                "Woman": "my parents screaming at me that video games cause violence",
                "Confused cat": "me building a house in Minecraft",
            }
        ),
    ),
]


def get_chat_completion(messages: List[Message]) -> str:
    result = openai.ChatCompletion.create(
        model=settings.model, messages=[msg.dict() for msg in messages], temperature=settings.temperature
    )
    return result["choices"][0]["message"]["content"]


def get_meme_template(scenario: str) -> TEMPLATES:
    template_name = get_chat_completion([*BASE_TEMPLATE_MESSAGES, Message(role=Role.USER, content=scenario)])
    return NAME_TO_TEMPLATE[template_name]


def get_meme_captions(meme_request: MemeRequest) -> Dict[str, str]:
    raw_captions = get_chat_completion([*BASE_CAPTIONS_MESSAGES, Message(role=Role.USER, content=meme_request.json())])
    return json.loads(raw_captions)


def get_captioned_meme_url(template_id: int, captions: List[str]) -> str:
    captions_params = {
        f'boxes[{i}][text]': text
        for i, text in enumerate(captions)
    }
    params = {
        "template_id": template_id,
        'username': settings.imgflip_username,
        'password': settings.imgflip_password,
        **captions_params,
    }
    response = requests.post(
        'https://api.imgflip.com/caption_image',
        data=params,
    )
    response.raise_for_status()
    return response.json()['data']['url']


def main():
    scenario = input("Enter Scenario: ")
    template = get_meme_template(scenario)
    meme_request = MemeRequest(scenario=scenario, template=template)
    captions = get_meme_captions(meme_request)
    print(f"Template: {template.name}")
    for name, content in captions.items():
        print(f"{name}: {content}")

    url = get_captioned_meme_url(template_id=template.id, captions=list(captions.values()))
    print(url)
    webbrowser.open(url)


if __name__ == "__main__":
    main()
