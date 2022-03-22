# NLU Lab Report

Mohamed Zarzoura

## Task 1: Implement first version

You will find training data under "01-Task_03UtterancesTrain" folder.

## Task 2: Evaluate NLU Exploratively

1. Feeding utterance from the training data:\
"What are the restaurants near me?" in `find_restaurent` is feed. Rasa select `find_restaurant` intention, which is correct, with relatively high confendence value of 0.7. The other intents ranking was as follows:
    1. `contact_police`, with confedence of 0.127
    2. `book_taxi`, with confedence of 0.114

2. Feeding utterance that are not the in training data:\
"Hello, I am looking for an open restaurant near here?" is feed. Rasa select `contact_police` intention, which is not correct, with relatively high confendence value of 0.438. The confedence value is descrease from the prevouis step. The other intents ranking was as follows:
    1. `find_restaurent`, surprisely with the lowest confedence value of 0.259 ---This should be the right choice.
    2. `book_taxi`, with confedence of 0.303

## Task 3: Evaluate NLU with test data

You will find "intent_report.json" file under "03-Task_03UtterancesTest" folder.

## Task 4: Improve NLU performance

You will find "intent_report.json" file and new data file under "04-Task_10UtterancesTest" folder.

The old weighted average F1 score was 0.423 and it became 0.966.

Generaly speaking I did a rough calculation of IT-IDF and found that words: "looking, town, need, food, expensive, cheap, moderately, returants, priced, need taxi, like, help, serves, book, want, police, station, restaurant" have the highest weight. I did this step to just get an idea about the words and subject that I need to focus on.

 For the `find_restaurent` intention, it seems to be that I need include information about price, place, serving food. I used *mainly* multiple varition of asking for information.

For the `book_taxi` intention, I included information about timing, places (from/to) and *mainly* multiple variations of asking for booking. It was noted that Timing and places keywords were not shown in the list above.

The `contact_police` was the hardest intention, there were not so may words that shown in the list. I included information about police station, some issues that needed to be handled like being robbed and *mainly* multiple variation of asking for contacting/finding the police.

## Task 5: Going deeper into confidence estimates

### Utterance 1: "no I don't want to contact the police"

Rasa choose the `contact_police` intention with confidence value of 0.769. The other intentions has a confidence value of 0.231.
May be rasa predict the need for contacting the police.

### Utterance 2: taxis are expensive, can I take public transport to Express by Holiday Inn Cambridge?

Rasa choose the `book_taxi` intention with confidence value of 0.612. The other intentions has a confidence value of 0.388.
May be rasa predict the asking for a taxi and the existence of a destination place!

### Utterance 3: what is a cryptocurrency?
Rasa choose the `find_restaurant` intention with confidence value of 0.599. The other intentions has a confidence value of 0.402.
May be rasa predict that here is a query about an information.

### Utterance 4: pratar du svenska?”

Rasa choose the `book_taxi` intention with confidence value of 0.713. The other intentions has a confidence value of 0.287.
I cannot figure out why the rasa choose this intention.