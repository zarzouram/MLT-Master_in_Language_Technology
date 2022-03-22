# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/core/actions/#custom-actions/


from typing import Any, Text, Dict, List
from rasa_sdk.events import SlotSet
import requests
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher


class ActionStateMachine(Action):

    def name(self) -> Text:
        return "action_state_machine"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        myintent = tracker.latest_message["intent"]["name"]
        personname = tracker.get_slot("person")
        meetingtime = tracker.get_slot("time")
        meetingdate = tracker.get_slot("date")

        infoencoded = [False if x=="/-" else True 
                        for x in [personname, meetingdate, meetingtime]]
        infoencoded = sum(v<<i for i, v in enumerate(infoencoded))

        if myintent == "inform":
            if infoencoded == 1:    # The user provided Person name only
                dispatcher.utter_message(template="utter_wh_date")
                return []

            # The user did not provide person name.
            elif infoencoded == 2 or infoencoded == 4 or+ infoencoded == 2 or infoencoded == 6:
                pass

            elif infoencoded == 3:    # The user provided person name and meeting date
                allday_event = tracker.get_slot("is_allday")
                if allday_event:    # may be not needed
                    dispatcher.utter_message(template="utter_inform_wo_time")
                    dispatcher.utter_message(template="utter_yn_correct")
                    return [SlotSet("status", "ask_yn_correct")]

                else:
                    dispatcher.utter_message(template="utter_is_allday")
                    return [SlotSet("status", "ask_is_allday")]

            elif infoencoded == 5:    # The user provided Person name and meeting time
                dispatcher.utter_message(template="utter_wh_date")
                return []
            
            elif infoencoded == 7:    # The user provided all information
                dispatcher.utter_message(template="utter_inform_all")
                dispatcher.utter_message(template="utter_yn_correct")
                return [SlotSet("status", "ask_yn_correct")]

        elif myintent == "affirm" or myintent == "deny":
            bot_status = tracker.get_slot("status")

            if not bot_status == "notdefined":
                if myintent == "affirm":

                    counter = 0
                    for event in reversed(tracker.events):
                        if not event["event"] == "action":
                            continue
                        else:
                            counter += 1
                        
                        if counter == 2:
                            if event['name'] == "utter_yn_correct":
                                bot_status = "ask_yn_correct"
                            else:
                                break

                    return [SlotSet("status", bot_status),
                            SlotSet("is_finished", True)]
                
                elif myintent == "deny" and bot_status == "ask_is_allday":
                    return [SlotSet("status", "ask_is_allday")]
                
                elif myintent == "deny" and bot_status == "ask_yn_correct":
                    return [SlotSet("person", "/-"), 
                            SlotSet("time", "/-"),
                            SlotSet("date", "/-"),
                            SlotSet("is_person_known", "notdefined"),
                            SlotSet("is_allday", False),
                            SlotSet("is_finished", False),
                            SlotSet("status", "notdefined")]
        
        else: return []