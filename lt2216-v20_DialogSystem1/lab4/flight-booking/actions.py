# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/core/actions/#custom-actions/


# This is a simple example for a custom action which utters "Hello World!"

# from typing import Any, Text, Dict, List
#
# from rasa_sdk import Action, Tracker
# from rasa_sdk.executor import CollectingDispatcher
#
#
# class ActionHelloWorld(Action):
#
#     def name(self) -> Text:
#         return "action_hello_world"
#
#     def run(self, dispatcher: CollectingDispatcher,
#             tracker: Tracker,
#             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
#
#         dispatcher.utter_message(text="Hello World!")
#
#         return []
from typing import Dict, Text, Any, List, Union, Optional

from rasa_sdk import Tracker, Action
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.forms import FormAction
from rasa_sdk.events import SlotSet


class FlightForm(FormAction):
    """Example of a custom form action"""

    def name(self) -> Text:
        """Unique identifier of the form"""

        return "flight_form"

    @staticmethod
    def required_slots(tracker: Tracker) -> List[Text]:
        """A list of required slots that the form has to fill"""
        source = tracker.get_slot("from")
        distination = tracker.get_slot("to")
        traveldate = tracker.get_slot("date")
        returndate = tracker.get_slot("return")
        travellclass = tracker.get_slot("class")
        oneway = tracker.get_slot("oneway")

        checkfromto = bool(source and distination)
        checkpath = bool(checkfromto and traveldate and travellclass)
        if oneway and checkpath:
            return ["from", "to", "date", "class", "oneway"]
        elif returndate:
            return ["from", "to", "date", "class", "return"]
        else:
            return ["from", "to", "date", "class", "oneway", "return"]
        
    def slot_mappings(self) -> Dict[Text, Union[Dict, List[Dict]]]:
        return {
            "oneway": [
                self.from_intent(intent="affirm", value=True),
                self.from_intent(intent="deny", value=False)
            ]
        }

    def validate_from(
            self,
            value: Text,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any],
        ) -> Dict[Text, Any]:

        distination = str(tracker.get_slot("to")).lower()
        if value.lower() == distination:
            dispatcher.utter_message(text="Departure canot be the same as distination")
            return {"from": None}
        else:
            return {"from": value}

    def validate_to(
            self,
            value: Text,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any],
        ) -> Dict[Text, Any]:
        
        source = str(tracker.get_slot("from")).lower()
        if value.lower() == source:
            dispatcher.utter_message(text="Distination canot be the same as departure")
            return {"to": None}
        else:
            return {"to": value}

    def submit(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict]:
        """Define what the form has to do
            after all required slots are filled"""
        returndate = bool(tracker.get_slot("return"))
        # utter submit template
        if returndate:
            dispatcher.utter_message(template="utter_inform1")
        else:
            dispatcher.utter_message(template="utter_inform")

        return []

class ActionCSlotsReset(Action):
   def name(self) -> Text:
      return "action_slots_reset"

   def run(self,
           dispatcher: CollectingDispatcher,
           tracker: Tracker,
           domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

      return [SlotSet("from", None), SlotSet("to", None), SlotSet("date", None), SlotSet("return", None), SlotSet("class", None), SlotSet("oneway", None)]
