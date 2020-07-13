# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/core/actions/#custom-actions/


# This is a simple example for a custom action which utters "Hello World!"

from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet

import random

JOKES = [("Doctor", "🤪😂"),
         ("Deja", "Knock-knock")]


class ActionJokeSetup(Action):

    def name(self) -> Text:
        return "action_joke_setup"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        jix = random.choice(range(len(JOKES)))

        dispatcher.utter_message(text=JOKES[jix][0])

        return [SlotSet("jix", jix)]


class ActionJokePunchline(Action):

    def name(self) -> Text:
        return "action_joke_punchline"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        jix = int(tracker.get_slot("jix"))
        joke = JOKES[jix]
        dispatcher.utter_message(text=joke[1])

        return []
