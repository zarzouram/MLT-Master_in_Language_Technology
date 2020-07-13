## Greeting
* greet
  - utter_welcomemessage

## Initialization: Bot start playing path
* affirm
  - utter_continuemessage
  - action_wait
  - utter_chooseplayer
* choose_a_player
  - action_who_start
  - slot{"current_player": "Bot", "fallback_status": "0"}
  - action_game_processing
  - slot{"current_player": "You", "fallback_status": "0"}
  - utter_entity_turn
  - action_wait
  - utter_choose_letter
  
## Initialization: Human start playing path 
* affirm
  - utter_continuemessage
  - action_wait
  - utter_chooseplayer
* choose_a_player
  - action_who_start
  - slot{"current_player": "You", "fallback_status": "0"}
  - utter_choose_letter

## Playing Game: Human Turn
* play_ghost
  - action_game_processing
  - slot{"current_player": "You", "fallback_status": "0"}
  - utter_entity_turn
  - action_wait
  - utter_choose_letter

## Playing Game: Bot Turn
* play_ghost
  - action_game_processing
  - slot{"current_player": "Bot", "fallback_status": "0"}
  - utter_entity_turn
  - action_game_processing
  - slot{"current_player": "You", "fallback_status": "0"}
  - utter_entity_turn
  - action_wait
  - utter_choose_letter

## termatation path
* greet
  - utter_welcomemessage
* deny
  - utter_goodbye

## say goodbye
* goodbye
  - utter_goodbye