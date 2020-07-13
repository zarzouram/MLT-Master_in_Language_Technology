## joke path
* ask_joke
  - utter_knockknock

<!-- ## deja vu path -->
<!-- * ask_joke -->
<!--   - utter_knockknock -->
<!-- * whoisthere -->
<!--   - utter_deja -->
<!-- * who -->
<!--   - utter_knockknock -->


<!-- ## doctor path -->
<!-- * ask_joke -->
<!--   - utter_knockknock -->
<!-- * whoisthere -->
<!--   - utter_doctor -->
<!-- * who -->
<!--   - utter_laugh -->

## happy path
* greet
  - utter_greet
* mood_great
  - utter_happy

## sad path 1
* greet
  - utter_greet
* mood_unhappy
  - utter_cheer_up
  - utter_did_that_help
* affirm
  - utter_happy

## sad path 2
* greet
  - utter_greet
* mood_unhappy
  - utter_cheer_up
  - utter_did_that_help
* deny
  - utter_goodbye

## say goodbye
* goodbye
  - utter_goodbye

## bot challenge
* bot_challenge
  - utter_iamabot

