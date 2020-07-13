## time path
* greet
  - utter_greet
* affirm 
  - utter_wh_person


## inform
* inform_person{"person":"Vlad Maraev"}
  - utter_wi_date
* inform_date{"date":"Tuesday"}
  - utter_wa_time
* inform_time{"time":"4" }
  - utter_inform
  
## incorrect
* deny
  - slot{"person": null}
  - utter_wh_person
