## time path
* greet
  - utter_greet
  - utter_wh_person

## deny_allday_event
* deny
  - action_state_machine
  - slot{"status": "ask_is_allday"}
  - utter_wh_time

## incorrect
* deny
  - action_state_machine
  - slot{"person": "/-", "time": "/-", "date": "/-", "is_person_known": "notdefined", "is_allday": 0, "is_finished": 0, "status": "notdefined"}
  - utter_wh_person

## affirm_to_finish_w_allday_event:
* affirm
  - action_state_machine
  - slot{"status": "ask_is_allday", "is_finished": "1"}
  - utter_inform_wo_time
  - utter_yn_correct

## affirm_to_finish:
* affirm
  - action_state_machine
  - slot{"status": "ask_yn_correct", "is_finished": "1"}
  - utter_allset

