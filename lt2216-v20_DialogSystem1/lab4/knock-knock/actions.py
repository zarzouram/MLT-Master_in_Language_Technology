# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/core/actions/#custom-actions/


# This is a simple example for a custom action which utters "Hello World!"

from typing import Any, Text, Dict, List
import requests
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet
import random


class ActionStartJock(Action):

    def name(self) -> Text:
        return "action_start_joke"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        dispatcher.utter_message(template="utter_knockknock")

        return [SlotSet("sessionstarted", True)]

class ActionJokeSetup(Action):

    def name(self) -> Text:
        return "action_joke_setup"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        jokesavail = tracker.get_slot("jokes")

        jid = random.choice(range(len(jokesavail)))
        thejoke = str(list(jokesavail[jid].keys())[0])

        dispatcher.utter_message(text=thejoke)

        return [SlotSet("jix", jid)]


class ActionJokePunchline(Action):

    def name(self) -> Text:
        return "action_joke_punchline"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        jid = int(tracker.get_slot("jix"))
        jokesavail = tracker.get_slot("jokes")
        jokesused = tracker.get_slot("jokes_used")

        thejokesetup = jokesavail.pop(jid)
        jokesused.append(thejokesetup)
        thepunchline = str(list(thejokesetup.values())[0])

        dispatcher.utter_message(text=thepunchline)

        if not jokesavail:  # repeat the jokes.
            jokesavail = jokesused[:]
            jokesused = []

        return [SlotSet("jokes", jokesavail), SlotSet("jokes_used", jokesused)]

class CheckSessionStarted(Action):
    def name(self):
        return "action_checksessionstarted"

    def run(self, dispatcher, tracker, domain):
        flag = bool(tracker.get_slot("sessionstarted"))
        return [SlotSet("sessionstarted", flag)]