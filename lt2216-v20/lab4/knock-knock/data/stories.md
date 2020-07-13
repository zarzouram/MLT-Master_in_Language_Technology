<!-- ## ask for a joke
* ask_joke
  - action_start_joke -->

# happy path 1
* mood_great
  - action_checksessionstarted
  - slot{"sessionstarted" : "1"}
  - utter_happy
  - utter_anotherjoke
* affirm
  - utter_knockknock

## happy path 2
* mood_great
  - action_checksessionstarted
  - slot{"sessionstarted" : "1"}
  - utter_happy
  - utter_anotherjoke
* deny
  - utter_goodbye

## sad path 1
* mood_unhappy
  - action_checksessionstarted
  - slot{"sessionstarted" : "1"}
  - utter_sorry
  - utter_blamedevloper
  - utter_anotherjoke
* affirm
  - utter_knockknock

## sad path 1
* mood_unhappy
  - action_checksessionstarted
  - slot{"sessionstarted" : "1"}
  - utter_sorry
  - utter_blamedevloper
  - utter_anotherjoke
* deny
  - utter_goodbye

## say goodbye
* goodbye
  - utter_goodbye
