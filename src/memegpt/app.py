import json
from enum import Enum
from typing import List, Dict

import openai
from pydantic import BaseModel, BaseSettings


class Settings(BaseSettings):
    openai_api_key: str
    temperature: int = 1
    model: str = "gpt-3.5-turbo"


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Message(BaseModel):
    role: Role
    content: str


class Template(BaseModel):
    name: str
    caption_names: List[str]


class MemeRequest(BaseModel):
    scenario: str
    template: Template


settings = Settings()

openai.api_key = settings.openai_api_key

TEMPLATES = [
    Template(name="distracted boyfriend", caption_names=["the boyfriend", "the ex", "the jealous girlfriend"]),
    Template(name="they're the same picture", caption_names=["left picture", "right picture"]),
    Template(name="drake hotline bling", caption_names=["bad", "good"]),
    Template(name="woman yelling at cat", caption_names=["woman", "cat"]),
    Template(name="expanding brain", caption_names=["small brain", "medium brain", "big brain"]),
]

NAME_TO_TEMPLATE = {template.name: template for template in TEMPLATES}
JOINED_TEMPLATE_NAMES = ", ".join(f"'{template}'" for template in TEMPLATES)

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
    Message(role=Role.ASSISTANT, content="woman yelling at cat"),
    Message(
        role=Role.USER,
        content=(
            "All those dumb employers are not willing to pay an extra 20K to give their good employees a raise. "
            "So then the good workers quit and the employers end up paying 50K extra to hire new workers."
        ),
    ),
    Message(role=Role.ASSISTANT, content="drake hotline bling"),
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
                template=NAME_TO_TEMPLATE["woman yelling at cat"],
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
            dict(
                woman="my parents screaming at me that video games cause violence",
                cat="me building a house in Minecraft",
            )
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


def main():
    scenario = input("Enter Scenario: ")
    template = get_meme_template(scenario)
    meme_request = MemeRequest(scenario=scenario, template=template)
    captions = get_meme_captions(meme_request)
    print(f"Template: {template.name}")
    for name, content in captions.items():
        print(f"{name}: {content}")


if __name__ == "__main__":
    main()
