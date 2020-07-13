<!-- ## time path -->
<!-- * greet -->
<!--   - utter_greet -->
<!--   - utter_wh_person -->

<!-- ## inform -->
<!-- * inform_person{"person":"Vlad Maraev"} -->
<!--   - utter_inform -->
<!--   - utter_yn_correct -->
  
<!-- ## incorrect -->
<!-- * deny -->
<!--   - slot{"person": null} -->
<!--   - utter_wh_person -->


## Deny path 1
* greet
  - utter_welcome_massege
* request_booking_flight
  - flight_form
  - form{"name": "flight_form"}
  - form{"name": null}
  - slot{"oneway": "0"}
* deny
  - utter_restrt
  - action_slots_reset
  - slot{"oneway": null}

## Deny path 2
* greet
  - utter_welcome_massege
* request_booking_flight
  - flight_form
  - form{"name": "flight_form"}
  - form{"name": null}
  - slot{"oneway": "1"}
* deny
  - utter_restrt
  - action_slots_reset
  - slot{"oneway": null}


## Deny path 3
* greet
  - utter_welcome_massege
* request_booking_flight
  - flight_form
  - form{"name": "flight_form"}
  - form{"name": null}
* deny
  - utter_restrt
  - action_slots_reset
  - slot{"oneway": null}

## Affirm path 1
* greet
  - utter_welcome_massege
* request_booking_flight
  - flight_form
  - form{"name": "flight_form"}
  - form{"name": null}
  - slot{"oneway": "0"}
* affirm
  - utter_submit

## Affirm path 2
* greet
  - utter_welcome_massege
* request_booking_flight
  - flight_form
  - form{"name": "flight_form"}
  - form{"name": null}
  - slot{"oneway": "1"}
* affirm
  - utter_submit

## Affirm path 3
* greet
  - utter_welcome_massege
* request_booking_flight
  - flight_form
  - form{"name": "flight_form"}
  - form{"name": null}
* affirm
  - utter_submit