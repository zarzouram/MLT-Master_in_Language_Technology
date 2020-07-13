# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/core/actions/#custom-actions/


# This is a simple example for a custom action which utters "Hello World!"

from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet, SessionStarted, ActionExecuted, EventType
from rasa_core_sdk.events import FollowupAction, UserUtteranceReverted

import requests
import random
from time import sleep

text = ["G", "H", "O", "S", "T"]

class ActionWhoStart(Action):

    def name(self) -> Text:
        return "action_who_start"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        userchoice = str(tracker.get_slot("numerical"))
        
        # couter = 0
        # get prevouis user intent
        previntent = ""
        for event in reversed(tracker.events):
            if event["event"] == "user":
                # couter+=1
                previntent = event["parse_data"]["intent"]["name"]
                # print("  ", previntent, end="\n\n")
                break
                # if couter == 2: break

        # This action function is called each time rasa predict the user intent is "choose_a_player", regardless of the story. The code checks if the intention prediction is inline with the designed story.
        # TODO: Another issue: if the user choice is correct but are written in words. Like "One" or "I will go for zero" rather than numerical input. The solution could be using dickling. However, I failed to setup ducling on server or my Windows machine.
        if (userchoice.isdigit() and not (previntent == "greet")):
            if (int(userchoice) == 0 or int(userchoice) == 1):
                
                # coin toss
                r = random.choice([0, 1])
                theplayer = "You" if int(userchoice) == r else "Bot"
                
                # Load dictionary and save it in a slot 
                #wordlistpath = "/usr/share/dict/words"
                wordlistpath = "/home/guszarzmo@GU.GU.SE/LT2216-v20/Project/development/wordslist.txt"    # sample list for testing purpose
                
                with open(wordlistpath, "r") as f:
                    wordlist = f.readlines()

                words = [word.rstrip().lower()
                         for word in wordlist if len(word.rstrip()) >= 5]

                player="Your" if theplayer == "You" else "My"
                dispatcher.utter_message(template="utter_entity_turn", current_player=player)
                sleep(0.7)
                
                return [SlotSet("current_player", theplayer), 
                        SlotSet("all_words", words), 
                        SlotSet("possible_words", words)]
        
        # Fallback if the intetion prediction is no inline with the story or user input is not correct
        elif not previntent == "":
            return [FollowupAction("action_custom_fallback"), 
                    SlotSet("fallback_status", "1")]
        else:
            return [FollowupAction("action_custom_fallback")]

class ActionGameProcessing(Action):

    def name(self) -> Text:
        return "action_game_processing"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        # couter = 0
        # get prevouis user intent
        previntent = ""
        for event in reversed(tracker.events):
            if event["event"] == "user":
                # couter+=1
                previntent = event["parse_data"]["intent"]["name"]
                break
                # print("  ", previntent, end="\n\n")
                # if couter == 2: break
        
        # This action function is called each time rasa predicts the user intent is "play_ghost", regardless of the story. The code checks if the intention prediction is inline with the designed story.
        if previntent == "choose_a_player" or previntent == "play_ghost" or previntent == "affirm":

            allwords = tracker.get_slot("all_words") # Dictionary
            # all possible words in the dictionary after players selections of letters
            possiblewords = tracker.get_slot("possible_words")
            # current part of word based on player selection of letters
            fragment = str(tracker.get_slot("fragment"))
            currentplayer = str(tracker.get_slot("current_player"))
            # the letter that have been selected by the human player
            myletter = str(tracker.get_slot("alphabet"))

            possiblewords = [
                word for word in possiblewords if word[:len(fragment)] == fragment]

            # get all possible next letters based on current word fragmant
            candidateletters = ""
            secondfragments = [word[len(fragment):][0] for word in possiblewords]
            for letter in secondfragments:
                if letter not in candidateletters:
                    candidateletters += letter

            if currentplayer == "You":
                # check if the choice is legal
                if myletter not in candidateletters:
                    dispatcher.utter_message(text="Illegal choice.")
                    sleep(0.7)
                    dispatcher.utter_message(template="utter_end_game", entity_loose="You", entity_win="I")
                    sleep(0.7)
                    
                    return [SlotSet("entity_loose", "You"), 
                            SlotSet("entity_win", "Bot"), 
                            FollowupAction("action_game_reset")]

            elif currentplayer == "Bot":
                # Bot choose a letter randoumly from candidateletters
                myletter = candidateletters[random.randint(
                    0, len(candidateletters) - 1)]
                message = "I pick: " + myletter
                dispatcher.utter_message(text=message)
                sleep(0.7)

            fragment += myletter
            message = "Word fragment: " + fragment
            dispatcher.utter_message(text=message)
            sleep(0.7)

            # check whether a word has been completed
            if fragment in allwords:
                entitywin = "Bot" if currentplayer == "You" else "You"
                dispatcher.utter_message(text="Word is compeleted.")
                sleep(0.7)
                dispatcher.utter_message(template="utter_end_game", entity_loose=currentplayer, entity_win=entitywin)
                sleep(0.7)
                
                return [SlotSet("entity_loose", currentplayer), 
                        SlotSet("entity_win", entitywin), 
                        FollowupAction("action_game_reset")]

            # next round
            currentplayer = "Bot" if currentplayer == "You" else "You"
            
            return [SlotSet("current_player", currentplayer), 
                    SlotSet("possible_words", possiblewords), 
                    SlotSet("fragment", fragment)]

        # Fallback if the intetion prediction is no inline with the story or user input is not correct
        elif not previntent == "":
            return [FollowupAction("action_custom_fallback"), 
                    SlotSet("fallback_status", "1")]
        else:
            return [FollowupAction("action_custom_fallback")]


class ActionGameReset(Action):

    def name(self) -> Text:
        return "action_game_reset"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        # reset the game or round

        botscore = tracker.get_slot("bot_score")
        humanscore = tracker.get_slot("human_score")
        entityloose = tracker.get_slot("entity_loose")
        entitywin = tracker.get_slot("entity_win")

        # Update the score of the  player who loses the game
        if entityloose == "You":
            humanscore += text[len(humanscore)]
        else:
            botscore += text[len(botscore)]
        
        dispatcher.utter_message(template="utter_scores", human_score=humanscore, bot_score=botscore)
        sleep(0.7)

        # check if game has ended or go for a next round 
        if len(botscore) == len(text) or len(humanscore) == len(text):
            dispatcher.utter_message(template="utter_end_ghost", entity_loose=entityloose, entity_win=entitywin)
            sleep(0.7)
            dispatcher.utter_message(template="utter_start_again")
            sleep(0.7)
            
            return [SlotSet("alphabet", ""), SlotSet("numerical", None), 
                    SlotSet("current_player", None), 
                    SlotSet("fallback_status", "0"), 
                    SlotSet("all_words", None), SlotSet("fragment", ""), 
                    SlotSet("possible_words", ""), 
                    SlotSet("entity_loose", None), SlotSet("entity_win", None),SlotSet("bot_score", ""), SlotSet("human_score", ""), FollowupAction("action_listen")]
        else:
            dispatcher.utter_message(template="utter_another_round")
            sleep(0.7)

            return [SlotSet("alphabet", ""), SlotSet("numerical", None), 
                    SlotSet("current_player", None), 
                    SlotSet("fallback_status", "0"), 
                    SlotSet("all_words", None), SlotSet("fragment", ""), SlotSet("possible_words", ""), 
                    SlotSet("entity_loose", None), SlotSet("entity_win", None),
                    SlotSet("bot_score", botscore), 
                    SlotSet("human_score", humanscore), FollowupAction("action_listen")]

class ActionCustomFallback(Action):

    def name(self) -> Text:
        return "action_custom_fallback"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        # the bot utterances depend on the fallback status:
        # fallbck status = 1 if the predicted user intention is not inline with the story otherwise it 0
        status = tracker.get_slot("fallback_status")
        myintent = tracker.latest_message["intent"]["name"]
        
        if status == "0":
            if myintent == "choose_a_player":
                dispatcher.utter_message(template="utter_wrong_choice")
                sleep(0.7)
                dispatcher.utter_message(template="utter_chooseplayer")
                sleep(0.7)

            elif myintent == "play_ghost":
                userinput = tracker.latest_message["text"]
                if not len(userinput) == 1:
                    dispatcher.utter_message(template="utter_wrong_letter")
                    sleep(0.7)
        else:
            dispatcher.utter_message(template="utter_default_fallback")
            sleep(0.7)

        # Utter the last bot message.
        for event in reversed(tracker.events):
            if event["event"] == "action":
                if event["name"].startswith("utter_"):
                    lastbotmessage = event["name"]
                    break
        
        dispatcher.utter_message(template=lastbotmessage)
        sleep(0.7)
        
        return [UserUtteranceReverted()]


class ActionSleep(Action):

    def name(self) -> Text:
        return "action_wait"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        sleep(0.7)

        return []
